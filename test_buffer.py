"""CUDA self-tests for :mod:`LCT.tensor_buffer.TensorBuffer`."""

from random import Random

import torch

from LCT.tensor_buffer import (
    Allocation,
    TensorBuffer,
    _ALIGNMENT,
    _STATUS_FREED,
    _STATUS_FREE_LIST_FULL,
    _STATUS_INVALID_FREE,
    _STATUS_INVALID_REQUEST,
    _STATUS_OK,
    _STATUS_OUT_OF_MEMORY,
    verify_buffer,
    visualize_buffer,
)


def _check(allocation: Allocation) -> tuple[int, int, int]:
    values = allocation.descriptor.cpu().tolist()
    return tuple(int(value) for value in values[:3])


def _request(buffer: TensorBuffer, nbytes: int) -> Allocation:
    return buffer.allocate(
        torch.tensor([nbytes], dtype=torch.int32, device=buffer.device)
    )


def _free_status(buffer: TensorBuffer, allocation: Allocation) -> int:
    return int(buffer.free(allocation).cpu().item())


def _free_model(
    free_regions: list[list[int]],
    offset: int,
    size: int,
) -> list[list[int]]:
    free_regions.append([offset, size])
    free_regions.sort()
    merged: list[list[int]] = []
    for start, length in free_regions:
        if merged and merged[-1][0] + merged[-1][1] == start:
            merged[-1][1] += length
        else:
            merged.append([start, length])
    return merged


def _show_buffer(label: str, buffer: TensorBuffer) -> None:
    print(f"\n{label}\n{visualize_buffer(buffer, width=48)}")


def _test_first_fit_and_coalescing() -> None:
    buffer = TensorBuffer(1024, max_free_regions=16)
    first = _request(buffer, 128)
    second = _request(buffer, 256)
    third = _request(buffer, 128)
    assert _check(first) == (0, 128, _STATUS_OK)
    assert _check(second) == (128, 256, _STATUS_OK)
    assert _check(third) == (384, 128, _STATUS_OK)
    verify_buffer(buffer)
    _show_buffer("Initial allocations", buffer)

    assert _free_status(buffer, second) == _STATUS_OK
    reused = _request(buffer, 64)
    assert _check(reused) == (128, 64, _STATUS_OK)
    _show_buffer("Earliest-fit reuse after fragmentation", buffer)

    assert _free_status(buffer, first) == _STATUS_OK
    earliest = _request(buffer, 128)
    assert _check(earliest) == (0, 128, _STATUS_OK)

    for allocation in (third, reused, earliest):
        assert _free_status(buffer, allocation) == _STATUS_OK
    verify_buffer(buffer)
    _show_buffer("Coalesced free space", buffer)
    assert _check(_request(buffer, 1024)) == (0, 1024, _STATUS_OK)


def _test_alignment_errors_and_reset() -> None:
    buffer = TensorBuffer(64, max_free_regions=8)
    first = _request(buffer, 1)
    second = _request(buffer, 17)
    assert _check(first) == (0, 16, _STATUS_OK)
    assert _check(second) == (16, 32, _STATUS_OK)
    assert _check(_request(buffer, 17)) == (-1, 0, _STATUS_OUT_OF_MEMORY)

    invalid = _request(buffer, 0)
    assert _check(invalid) == (-1, 0, _STATUS_INVALID_REQUEST)
    assert _free_status(buffer, invalid) == _STATUS_INVALID_FREE

    assert _free_status(buffer, first) == _STATUS_OK
    assert int(first.status.cpu().item()) == _STATUS_FREED
    assert _free_status(buffer, first) == _STATUS_INVALID_FREE
    verify_buffer(buffer)

    buffer.reset()
    assert _free_status(buffer, second) == _STATUS_INVALID_FREE
    verify_buffer(buffer)
    assert _check(_request(buffer, 64)) == (0, 64, _STATUS_OK)


def _test_free_list_limit_and_ownership() -> None:
    buffer = TensorBuffer(96, max_free_regions=2)
    allocations = [_request(buffer, 16) for _ in range(6)]
    assert all(_check(allocation)[2] == _STATUS_OK for allocation in allocations)

    assert _free_status(buffer, allocations[0]) == _STATUS_OK
    assert _free_status(buffer, allocations[2]) == _STATUS_OK
    assert _free_status(buffer, allocations[4]) == _STATUS_FREE_LIST_FULL
    verify_buffer(buffer)
    _show_buffer("Free-list metadata limit", buffer)

    assert _free_status(buffer, allocations[1]) == _STATUS_OK
    assert _free_status(buffer, allocations[4]) == _STATUS_OK
    assert _free_status(buffer, allocations[3]) == _STATUS_OK
    assert _free_status(buffer, allocations[5]) == _STATUS_OK
    verify_buffer(buffer)
    _show_buffer("Free-list space recovered by coalescing", buffer)
    assert _check(_request(buffer, 96)) == (0, 96, _STATUS_OK)

    first = TensorBuffer(32, max_free_regions=4)
    second = TensorBuffer(32, max_free_regions=4)
    allocation = _request(first, 16)
    try:
        second.free(allocation)
    except ValueError:
        pass
    else:
        raise AssertionError("cross-buffer free should be rejected")


def _test_randomized_first_fit() -> None:
    capacity = 4096
    buffer = TensorBuffer(capacity, max_free_regions=64)
    free_regions = [[0, capacity]]
    active: list[tuple[Allocation, int, int]] = []
    rng = Random(0)

    for _ in range(200):
        if active and rng.random() < 0.45:
            index = rng.randrange(len(active))
            allocation, offset, size = active.pop(index)
            assert _free_status(buffer, allocation) == _STATUS_OK
            free_regions = _free_model(free_regions, offset, size)
        else:
            request = rng.randrange(1, 257)
            size = (request + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
            candidate = next(
                (
                    (index, region)
                    for index, region in enumerate(free_regions)
                    if region[1] >= size
                ),
                None,
            )
            allocation = _request(buffer, request)
            offset, allocated, status = _check(allocation)
            if candidate is None:
                assert (offset, allocated, status) == (-1, 0, _STATUS_OUT_OF_MEMORY)
            else:
                region_index, region = candidate
                assert (offset, allocated, status) == (
                    region[0],
                    size,
                    _STATUS_OK,
                )
                region[0] += size
                region[1] -= size
                if region[1] == 0:
                    free_regions.pop(region_index)
                active.append((allocation, offset, allocated))
        verify_buffer(buffer)

    for allocation, offset, size in active:
        assert _free_status(buffer, allocation) == _STATUS_OK
        free_regions = _free_model(free_regions, offset, size)
    verify_buffer(buffer)
    _show_buffer("Randomized test after full coalescing", buffer)
    assert _check(_request(buffer, capacity)) == (0, capacity, _STATUS_OK)


def main() -> None:
    _test_first_fit_and_coalescing()
    _test_alignment_errors_and_reset()
    _test_free_list_limit_and_ownership()
    _test_randomized_first_fit()
    print("GPU first-fit allocator self-tests passed")


if __name__ == "__main__":
    main()
