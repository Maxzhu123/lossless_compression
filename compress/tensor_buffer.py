"""GPU-resident first-fit allocator for descriptor-based CUDA buffers."""

from dataclasses import dataclass, fields
import torch
import triton
from triton import language as tl


_ALIGNMENT = 16
_STATUS_OK = 0
_STATUS_OUT_OF_MEMORY = 1
_STATUS_INVALID_REQUEST = 2
_STATUS_FREE_LIST_FULL = 3
_STATUS_INVALID_FREE = 4
_STATUS_FREED = 5


@triton.jit
def _lock(lock):
    while tl.atomic_cas(lock, 0, 1) != 0:
        pass


@triton.jit
def _unlock(lock):
    tl.atomic_xchg(lock, 0)


@triton.jit
def _reset_kernel(
    free_starts,
    free_sizes,
    free_count,
    lock,
    generation,
    capacity_bytes,
    MAX_FREE_REGIONS: tl.constexpr,
):
    indices = tl.arange(0, MAX_FREE_REGIONS)
    tl.store(free_starts + indices, 0)
    tl.store(free_sizes + indices, 0)
    tl.store(free_sizes, capacity_bytes)
    tl.store(free_count, 1)
    tl.store(lock, 0)
    tl.store(generation, tl.load(generation) + 1)


@triton.jit
def _allocate_kernel(
    requested_bytes,
    descriptor,
    free_starts,
    free_sizes,
    free_count,
    lock,
    generation,
    ALIGNMENT: tl.constexpr,
    MAX_FREE_REGIONS: tl.constexpr,
):
    _lock(lock)

    request = tl.load(requested_bytes).to(tl.int32)
    current_generation = tl.load(generation).to(tl.int32)
    size = ((request + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    count = tl.load(free_count).to(tl.int32)
    indices = tl.arange(0, MAX_FREE_REGIONS)
    active = indices < count
    starts = tl.load(free_starts + indices, mask=active, other=0)
    sizes = tl.load(free_sizes + indices, mask=active, other=0)
    candidates = tl.where(active & (sizes >= size), indices, MAX_FREE_REGIONS)
    slot = tl.min(candidates, axis=0)
    success = (request > 0) & (slot < count)

    start = tl.load(free_starts + slot, mask=success, other=0)
    region_size = tl.load(free_sizes + slot, mask=success, other=0)
    exact = success & (region_size == size)
    partial = success & ~exact

    next_indices = indices + 1
    shift_left = exact & (indices >= slot) & (next_indices < count)
    next_starts = tl.load(free_starts + next_indices, mask=shift_left, other=0)
    next_sizes = tl.load(free_sizes + next_indices, mask=shift_left, other=0)
    tl.store(free_starts + indices, next_starts, mask=shift_left)
    tl.store(free_sizes + indices, next_sizes, mask=shift_left)
    tl.store(free_count, count - 1, mask=exact)

    tl.store(free_starts + slot, start + size, mask=partial)
    tl.store(free_sizes + slot, region_size - size, mask=partial)
    tl.store(descriptor, tl.where(success, start, -1))
    tl.store(descriptor + 1, tl.where(success, size, 0))
    tl.store(
        descriptor + 2,
        tl.where(request <= 0, 2, tl.where(success, 0, 1)),
    )
    tl.store(descriptor + 3, current_generation)
    _unlock(lock)


@triton.jit
def _free_kernel(
    descriptor,
    free_starts,
    free_sizes,
    free_count,
    lock,
    generation,
    status_out,
    MAX_FREE_REGIONS: tl.constexpr,
):
    _lock(lock)

    offset = tl.load(descriptor).to(tl.int32)
    size = tl.load(descriptor + 1).to(tl.int32)
    allocation_status = tl.load(descriptor + 2).to(tl.int32)
    allocation_generation = tl.load(descriptor + 3).to(tl.int32)
    current_generation = tl.load(generation).to(tl.int32)
    count = tl.load(free_count).to(tl.int32)
    valid = (
        (allocation_status == 0)
        & (allocation_generation == current_generation)
        & (offset >= 0)
        & (size > 0)
    )

    indices = tl.arange(0, MAX_FREE_REGIONS)
    active = indices < count
    starts = tl.load(free_starts + indices, mask=active, other=0)
    sizes = tl.load(free_sizes + indices, mask=active, other=0)
    insert = tl.sum((active & (starts < offset)).to(tl.int32), axis=0)

    has_previous = insert > 0
    has_next = insert < count
    previous_index = insert - 1
    previous_start = tl.load(
        free_starts + previous_index, mask=has_previous, other=0
    )
    previous_size = tl.load(
        free_sizes + previous_index, mask=has_previous, other=0
    )
    next_start = tl.load(free_starts + insert, mask=has_next, other=0)
    next_size = tl.load(free_sizes + insert, mask=has_next, other=0)
    merge_previous = has_previous & (previous_start + previous_size == offset)
    merge_next = has_next & (offset + size == next_start)
    insert_new = valid & ~merge_previous & ~merge_next & (count < MAX_FREE_REGIONS)
    previous_only = valid & merge_previous & ~merge_next
    next_only = valid & ~merge_previous & merge_next
    both = valid & merge_previous & merge_next

    tl.store(free_sizes + previous_index, previous_size + size, mask=previous_only)
    tl.store(free_starts + insert, offset, mask=next_only)
    tl.store(free_sizes + insert, next_size + size, mask=next_only)
    tl.store(
        free_sizes + previous_index,
        previous_size + size + next_size,
        mask=both,
    )

    next_indices = indices + 1
    shift_left = both & (indices >= insert) & (next_indices < count)
    shifted_starts = tl.load(free_starts + next_indices, mask=shift_left, other=0)
    shifted_sizes = tl.load(free_sizes + next_indices, mask=shift_left, other=0)
    tl.store(free_starts + indices, shifted_starts, mask=shift_left)
    tl.store(free_sizes + indices, shifted_sizes, mask=shift_left)
    tl.store(free_count, count - 1, mask=both)

    shift_right = insert_new & (indices >= insert) & (indices < count)
    tl.store(free_starts + indices + 1, starts, mask=shift_right)
    tl.store(free_sizes + indices + 1, sizes, mask=shift_right)
    tl.store(free_starts + insert, offset, mask=insert_new)
    tl.store(free_sizes + insert, size, mask=insert_new)
    tl.store(free_count, count + 1, mask=insert_new)

    free_list_full = valid & ~merge_previous & ~merge_next & (
        count >= MAX_FREE_REGIONS
    )
    tl.store(descriptor + 2, 5, mask=valid & ~free_list_full)
    tl.store(
        status_out,
        tl.where(~valid, 4, tl.where(free_list_full, 3, 0)),
    )
    _unlock(lock)


@dataclass(frozen=True)
class Allocation:
    """A CUDA ``[offset, aligned_size, status, generation]`` descriptor."""

    descriptor: torch.Tensor
    owner: object

    @property
    def offset(self) -> torch.Tensor:
        return self.descriptor[0]

    @property
    def size(self) -> torch.Tensor:
        return self.descriptor[1]

    @property
    def status(self) -> torch.Tensor:
        return self.descriptor[2]

    @property
    def generation(self) -> torch.Tensor:
        return self.descriptor[3]


class TensorBuffer:
    """CUDA-resident first-fit allocator returning device-side descriptors.

    Callers pass the payload base and returned device offsets directly to
    kernels; dynamically sized Python tensor views are intentionally avoided.
    """

    def __init__(
        self, capacity_bytes: int, *,
        max_free_regions: int = 256,
        device: torch.device | str | None = None,
    ) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        if capacity_bytes > torch.iinfo(torch.int32).max:
            raise ValueError("capacity_bytes must fit in int32")
        if max_free_regions <= 0 or max_free_regions & (max_free_regions - 1):
            raise ValueError("max_free_regions must be a positive power of two")

        self.capacity_bytes = int(capacity_bytes)
        self.max_free_regions = int(max_free_regions)
        self.device = torch.device(device or "cuda")
        self._metadata_bytes = self._align(8 * self.max_free_regions + 12)
        self._storage = torch.zeros(
            self._metadata_bytes + self.capacity_bytes,
            dtype=torch.uint8,
            device=self.device,
        )
        self.device = self._storage.device
        self._data = self._storage.narrow(
            0, self._metadata_bytes, self.capacity_bytes
        )
        self._free_starts = self._storage.narrow(
            0, 0, self.max_free_regions * 4
        ).view(torch.int32)
        self._free_sizes = self._storage.narrow(
            0, self.max_free_regions * 4, self.max_free_regions * 4
        ).view(torch.int32)
        self._free_count = self._storage.narrow(
            0, self.max_free_regions * 8, 4
        ).view(torch.int32)
        self._lock = self._storage.narrow(
            0, self.max_free_regions * 8 + 4, 4
        ).view(torch.int32)
        self._generation = self._storage.narrow(
            0, self.max_free_regions * 8 + 8, 4
        ).view(torch.int32)
        self.reset()

    @staticmethod
    def _align(nbytes: int) -> int:
        return (nbytes + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT

    @property
    def data(self) -> torch.Tensor:
        return self._data

    def allocate(self, nbytes: torch.Tensor) -> Allocation:
        """Reserve a one-element CUDA integer request without host sync."""
        descriptor = torch.empty(4, dtype=torch.int32, device=self.device)
        _allocate_kernel[(1,)](
            nbytes, descriptor,
            self._free_starts, self._free_sizes, self._free_count, self._lock, self._generation,
            ALIGNMENT=_ALIGNMENT, MAX_FREE_REGIONS=self.max_free_regions,
        )
        return Allocation(descriptor, self)

    def free(self, allocation: Allocation) -> torch.Tensor:
        """Return an allocation and asynchronously return a device status."""
        if allocation.owner is not self:
            raise ValueError("allocation belongs to a different allocator")
        if allocation.descriptor.device != self.device:
            raise ValueError("allocation must be on the allocator device")
        status = torch.empty(1, dtype=torch.int32, device=self.device)
        _free_kernel[(1,)](
            allocation.descriptor,
            self._free_starts, self._free_sizes, self._free_count,
            self._lock, self._generation, status,
            MAX_FREE_REGIONS=self.max_free_regions,
        )
        return status

    def reset(self) -> None:
        """Reset asynchronously; previously returned descriptors become stale."""
        _reset_kernel[(1,)](
            self._free_starts, self._free_sizes,  self._free_count,
            self._lock, self._generation, self.capacity_bytes,
            MAX_FREE_REGIONS=self.max_free_regions,
        )


def _free_regions_snapshot(
    buffer: TensorBuffer,
) -> list[tuple[int, int]]:
    """Synchronize and copy the device allocator's free-list metadata."""
    metadata = buffer._storage.narrow(0, 0, buffer._metadata_bytes).view(
        torch.int32
    ).cpu()
    count = int(metadata[2 * buffer.max_free_regions].item())
    if not 0 <= count <= buffer.max_free_regions:
        raise RuntimeError(f"corrupt free-list count: {count}")
    return [
        (
            int(metadata[index].item()),
            int(metadata[buffer.max_free_regions + index].item()),
        )
        for index in range(count)
    ]


def verify_buffer(buffer: TensorBuffer) -> None:
    """Synchronize and validate the allocator's free-list invariants."""
    previous_end = 0
    free_bytes = 0
    for start, size in _free_regions_snapshot(buffer):
        if size <= 0 or start < previous_end or start + size > buffer.capacity_bytes:
            raise RuntimeError(f"invalid free region: [{start}, {start + size})")
        if start == previous_end and previous_end != 0:
            raise RuntimeError("adjacent free regions were not coalesced")
        previous_end = start + size
        free_bytes += size
    if free_bytes > buffer.capacity_bytes:
        raise RuntimeError("free regions exceed payload capacity")


def visualize_buffer(buffer: TensorBuffer, width: int = 80) -> str:
    """Synchronize and return an ASCII map of payload usage."""
    if width < 8:
        raise ValueError("width must be at least 8")
    free_regions = _free_regions_snapshot(buffer)
    free_bytes = sum(size for _, size in free_regions)
    used_bytes = buffer.capacity_bytes - free_bytes
    cells = ["#"] * width
    for start, size in free_regions:
        left = start * width // buffer.capacity_bytes
        right = (start + size) * width + buffer.capacity_bytes - 1
        right //= buffer.capacity_bytes
        for cell in range(left, min(width, max(left + 1, right))):
            cells[cell] = "."
    ranges = [f"[{start}, {start + size})" for start, size in free_regions]
    if len(ranges) > 8:
        ranges = [*ranges[:4], "...", *ranges[-4:]]
    return "\n".join(
        (
            f"payload: {used_bytes}/{buffer.capacity_bytes} B used "
            f"({used_bytes / buffer.capacity_bytes:.1%}); "
            f"{len(free_regions)} free region(s)",
            f"0 [{''.join(cells)}] {buffer.capacity_bytes}",
            "# used  . free",
            f"free: {', '.join(ranges) or 'none'}",
        )
    )
