import gc
import statistics
import torch
import torch.nn.functional as F


BATCH = 32
SEQ = 2048
HIDDEN = 4096
ITERS = 10
WARMUP = 3
DTYPE = torch.bfloat16
EPS = None
COMPILE_MANUAL = False


class ManualRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps=None):
        if eps is None:
            eps = torch.finfo(x.dtype).eps

        # input_dtype = x.dtype
        # x = x.float()
        # rstd = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
        #
        # ctx.save_for_backward(x, rstd)
        # ctx.n = x.shape[-1]
        #
        # return (x * rstd).to(input_dtype)
        y, rstd = torch.ops.aten._fused_rms_norm.default(
            x,
            [x.shape[-1]],
            None,  # weight
            eps,
        )
        ctx.save_for_backward(x, rstd)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, rstd = ctx.saved_tensors
        # x, grad_output = x.float(), grad_output.float()
        #
        # dot = (grad_output * x).sum(dim=-1, keepdim=True)
        #
        # grad_x = (
        #     grad_output * rstd
        #     - x * rstd.pow(3) * dot / n
        # )
        #
        # return grad_x.bfloat16(), None

        grad_x, grad_weight = torch.ops.aten._fused_rms_norm_backward.default(
            grad_output,
            x,
            [x.shape[-1]],
            rstd,
            None,  # weight
            [True, False],  # request grad_input, not grad_weight
        )
        return grad_x, None


def manual_rmsnorm(x, eps):
    x_norm = ManualRMSNorm.apply(x, eps)
    return x_norm


def pytorch_rmsnorm(x, eps):
    return F.rms_norm(x, (x.shape[-1],), weight=None, eps=eps)


def mib(nbytes):
    return nbytes / (1024 ** 2)


def cleanup_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def measure_memory(fn, x, eps, iters):
    """
    Measure CUDA peak memory for forward and backward separately.

    For each iteration:
      1. clean up prior tensors/cached blocks as much as practical
      2. run forward with autograd enabled
      3. record the forward peak while the graph is still live
      4. reset peak-memory counters
      5. run backward
      6. record the backward-only peak

    Returned deltas subtract the memory allocated before each pass.
    """
    forward_peak = []
    backward_peak = []

    for _ in range(iters):
        cleanup_cuda()

        x_input = x.detach().clone().requires_grad_()
        baseline_forward = torch.cuda.memory_allocated()

        torch.cuda.reset_peak_memory_stats()

        out = fn(x_input, eps)
        torch.cuda.synchronize()
        forward_peak.append(
            torch.cuda.max_memory_allocated() - baseline_forward
        )

        grad_output = torch.ones_like(out)
        torch.cuda.synchronize()
        baseline_backward = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        out.backward(grad_output)
        torch.cuda.synchronize()
        backward_peak.append(
            torch.cuda.max_memory_allocated() - baseline_backward
        )

        del x_input, out, grad_output

    return {
        "forward_peak": forward_peak,
        "backward_peak": backward_peak,
    }


def summarize(name, result):
    forward = statistics.mean(result["forward_peak"])
    backward = statistics.mean(result["backward_peak"])
    print(
        f"{name}: forward peak {mib(forward):.2f} MiB, "
        f"backward peak {mib(backward):.2f} MiB"
    )


def report_difference(name, actual, expected):
    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff / expected.float().abs().clamp_min(1e-12)).max().item()
    print(f"{name}: max abs {max_abs:.6g}, max rel {max_rel:.6g}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    dtype = DTYPE
    eps = torch.finfo(dtype).eps if EPS is None else EPS

    device = torch.device("cuda")

    # Inputs are allocated before measurement and therefore excluded from the
    # reported deltas.
    x = torch.randn(
        BATCH,
        SEQ,
        HIDDEN,
        device=device,
        dtype=dtype,
    )

    compiled_pytorch = torch.compile(pytorch_rmsnorm)
    eager_pytorch = pytorch_rmsnorm

    if COMPILE_MANUAL:
        manual_fn = torch.compile(manual_rmsnorm)
        manual_name = "Manual RMSNorm (torch.compile)"
    else:
        manual_fn = manual_rmsnorm
        manual_name = "Manual RMSNorm (eager)"

    # Compilation/workspace allocation is deliberately excluded from the
    # benchmark. First calls can include large one-time compiler allocations.
    with torch.no_grad():
        for _ in range(WARMUP):
            for fn in (manual_fn, eager_pytorch, compiled_pytorch):
                y = fn(x, eps)
                del y
        torch.cuda.synchronize()

    cleanup_cuda()

    with torch.enable_grad():
        x_manual = x.detach().clone().requires_grad_()
        x_eager = x.detach().clone().requires_grad_()
        x_pytorch = x.detach().clone().requires_grad_()

        y_manual = manual_fn(x_manual, eps)
        y_eager = eager_pytorch(x_eager, eps)
        y_pytorch = compiled_pytorch(x_pytorch, eps)
        torch.cuda.synchronize()

        report_difference("Forward (manual vs eager)", y_manual, y_eager)
        report_difference("Forward (eager vs compiled)", y_eager, y_pytorch)

        grad_output = torch.randn_like(y_manual)
        grads_manual = torch.autograd.grad(y_manual, x_manual, grad_output)
        grads_eager = torch.autograd.grad(y_eager, x_eager, grad_output)
        grads_pytorch = torch.autograd.grad(y_pytorch, x_pytorch, grad_output)

        report_difference(
            "Backward input gradient (manual vs eager)",
            grads_manual[0],
            grads_eager[0],
        )
        report_difference(
            "Backward input gradient (eager vs compiled)",
            grads_eager[0],
            grads_pytorch[0],
        )

        del (
            x_manual,
            x_eager,
            x_pytorch,
            y_manual,
            y_eager,
            y_pytorch,
            grad_output,
            grads_manual,
            grads_eager,
            grads_pytorch,
        )
    print()
    cleanup_cuda()
    manual_result = measure_memory(
        manual_fn, x, eps, ITERS
    )
    eager_result = measure_memory(
        eager_pytorch, x, eps, ITERS
    )
    pytorch_result = measure_memory(
        compiled_pytorch, x, eps, ITERS
    )

    summarize(manual_name, manual_result)
    summarize(
        "F.rms_norm (eager)",
        eager_result,
    )
    summarize(
        "F.rms_norm (torch.compile)",
        pytorch_result,
    )



if __name__ == "__main__":
    main()
