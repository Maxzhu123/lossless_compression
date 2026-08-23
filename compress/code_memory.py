"""Small persistent GPU memory buffer for the bfloat16 compressor.

This module provides a fixed-size GPU byte buffer with a first-fit free-list
allocator.  Allocations carve views out of the persistent ``uint8`` tensor.
Freed regions are returned to the free list and immediately coalesced with
neighbouring free regions, so the next allocation reuses the lowest available
byte range.

The allocator metadata is kept on the Python side, so operations do not need
to copy GPU state back to the host and therefore do not introduce device
synchronization.
"""

import bisect
import torch


class TensorBuffer:
    """A fixed-size persistent device byte buffer with first-fit reuse.

    The buffer is created once with a fixed capacity.  Allocations return a
    view into the shared byte tensor and a byte offset.  Calling :meth:`free`
    returns the region to the allocator; the next allocation takes the first
    free region that is large enough (first fit).
    """

    def __init__(
        self,
        capacity_bytes: int,
        device: torch.device | str | None = None,
    ) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self.capacity_bytes = int(capacity_bytes)
        self.device = torch.device(device or "cuda")
        self._data = torch.empty(
            self.capacity_bytes,
            dtype=torch.uint8,
            device=self.device,
        )
        # Free regions are [start_byte, size_bytes] in ascending start order.
        self._free = [[0, self.capacity_bytes]]
        # Parallel sorted list of free-region starts, for O(log n) lookup.
        self._free_starts = [0]
        # Active allocations: byte offset -> size in bytes.
        self._allocations: dict[int, int] = {}

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def used_bytes(self) -> int:
        return sum(self._allocations.values())

    @property
    def free_bytes(self) -> int:
        return self.capacity_bytes - self.used_bytes

    def allocate(
        self,
        nbytes: int,
        dtype: torch.dtype = torch.int8,
    ) -> tuple[torch.Tensor, int]:
        """Return a tensor view of ``nbytes`` and its byte offset.

        Uses first-fit placement: the first free region large enough is chosen.
        The returned tensor is a view into the persistent buffer, not a new
        GPU allocation.
        """
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        itemsize = dtype.itemsize
        if nbytes % itemsize:
            raise ValueError(
                f"nbytes={nbytes} is not divisible by {dtype} itemsize={itemsize}"
            )

        for i, (offset, size) in enumerate(self._free):
            if size < nbytes:
                continue
            self._free.pop(i)
            self._free_starts.pop(i)
            if size > nbytes:
                self._free.insert(i, [offset + nbytes, size - nbytes])
                self._free_starts.insert(i, offset + nbytes)
            self._allocations[offset] = nbytes
            byte_view = self._data.narrow(0, offset, nbytes)
            tensor = byte_view.view(dtype)
            return tensor, offset

        raise RuntimeError(
            f"TensorBuffer has no free region large enough: need {nbytes} bytes, "
            f"free {self.free_bytes} bytes"
        )

    def free(self, allocation: torch.Tensor | int) -> None:
        """Mark a region as deleted and return it to the free list.

        Accepts either the allocated tensor view or a byte offset.
        """
        if isinstance(allocation, torch.Tensor):
            offset = allocation.storage_offset() * allocation.element_size()
            nbytes = allocation.nbytes
        else:
            offset = int(allocation)
            nbytes = self._allocations.get(offset)
            if nbytes is None:
                raise ValueError(f"no active allocation at offset {offset}")

        active = self._allocations.pop(offset)
        if nbytes != active:
            raise ValueError(
                f"mismatched free size at offset {offset}: "
                f"recorded {active}, got {nbytes}"
            )

        new_seg = [offset, nbytes]
        # Insert in start-order; coalesce with the previous and next free
        # regions when they touch.
        idx = bisect.bisect_left(self._free_starts, offset)
        if idx > 0 and self._free[idx - 1][0] + self._free[idx - 1][1] == offset:
            prev = self._free.pop(idx - 1)
            self._free_starts.pop(idx - 1)
            prev[1] += nbytes
            new_seg = prev
            idx -= 1

        self._free.insert(idx, new_seg)
        self._free_starts.insert(idx, new_seg[0])

        if (
            idx + 1 < len(self._free)
            and new_seg[0] + new_seg[1] == self._free[idx + 1][0]
        ):
            new_seg[1] += self._free.pop(idx + 1)[1]
            self._free_starts.pop(idx + 1)

    def reset(self) -> None:
        """Release every allocation and reset the buffer to one free region."""
        self._free = [[0, self.capacity_bytes]]
        self._free_starts = [0]
        self._allocations.clear()
