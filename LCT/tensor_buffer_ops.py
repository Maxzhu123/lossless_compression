"""Opaque allocator operations for torch.compile."""
import torch


@torch.library.custom_op("lct::tensor_buffer_free", mutates_args=("storage", "descriptor"))
def tensor_buffer_free(storage: torch.Tensor, descriptor: torch.Tensor,
                       max_free_regions: int) -> torch.Tensor:
    from .tensor_buffer import _free_kernel

    # Pass the shared storage as one argument: the metadata views alias it.
    metadata = storage.narrow(0, 0, 8 * max_free_regions + 12).view(torch.int32)
    status = torch.empty(1, dtype=torch.int32, device=storage.device)
    _free_kernel[(1,)](
        descriptor,
        metadata[:max_free_regions], metadata[max_free_regions:2 * max_free_regions],
        metadata[2 * max_free_regions:2 * max_free_regions + 1],
        metadata[2 * max_free_regions + 1:2 * max_free_regions + 2],
        metadata[2 * max_free_regions + 2:2 * max_free_regions + 3],
        status, MAX_FREE_REGIONS=max_free_regions,
    )
    return status


@tensor_buffer_free.register_fake
def _tensor_buffer_free_fake(storage, descriptor, max_free_regions):
    return torch.empty(1, dtype=torch.int32, device=storage.device)
