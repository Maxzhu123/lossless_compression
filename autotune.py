import torch
import tilelang
import tilelang.language as T


configs = [
    {"block": 128},
    {"block": 256},
    {"block": 512},
]


@tilelang.autotune(
    configs=configs,
    warmup=5,
    rep=20,
    skip_check=True,
)
@tilelang.jit
def vector_add(
    A,
    B,
    block: int = 256,
):
    N = T.const("N")

    A: T.Tensor((N,), T.float32)
    B: T.Tensor((N,), T.float32)
    C = T.empty((N,), T.float32)

    with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
        for tx in T.Parallel(block):
            i = bx * block + tx
            if i < N:
                C[i] = A[i] + B[i]

    return C


a = torch.randn(4096, device="cuda")
b = torch.randn(4096, device="cuda")

c = vector_add(a, b)

torch.testing.assert_close(c, a + b)
