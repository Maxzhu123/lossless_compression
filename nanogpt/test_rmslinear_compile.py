"""Standalone CUDA reproducer for _RMSNormQKV; run directly in PyCharm.

Compare compressed and dense runs using the switches below. Graph code shows
what Dynamo captures. Search for rms_qkv and rms_qkv_backward: explicitly
compiled backward helpers can appear as AOT Forward graphs.
"""
import torch
import torch.nn.functional as F

from LCT.layers import Linear, _RMSNormQKV
from LCT.dist_configs import act_dist
from LCT.tensor_buffer import TensorBuffer

COMPILE = True  # Outer wrapper; QKV's internal helpers are independently compiled.
COMPRESS_ACTIVATIONS = True
COMPRESS_WEIGHTS = False
BUFFER = True
LOG_GRAPH_BREAKS = True
LOG_GRAPH_CODE = True
LOG_AOT_GRAPHS = True
STEPS = 3
BATCH, SEQ_LEN, WIDTH, HIDDEN = 2, 64, 768, 768


def main():
    torch.manual_seed(0)
    torch._logging.set_logs(graph_breaks=LOG_GRAPH_BREAKS,
                            graph_code=LOG_GRAPH_CODE, aot_graphs=LOG_AOT_GRAPHS,
                            recompiles=True)
    print(f"PyTorch {torch.__version__}: compile={COMPILE}, "
          f"activations={COMPRESS_ACTIVATIONS}, weights={COMPRESS_WEIGHTS}, "
          f"buffer={BUFFER}", flush=True)

    x = torch.randn(BATCH, SEQ_LEN, WIDTH, device="cuda",
                    dtype=torch.bfloat16, requires_grad=True)
    layers = [Linear(WIDTH, HIDDEN).cuda() for _ in range(3)]
    gain = torch.randn(WIDTH, device="cuda", requires_grad=True)
    buffer = TensorBuffer(8 * 2**20, device="cuda") if BUFFER else None
    inputs = (x, gain, *(p for layer in layers for p in (layer.weight, layer.bias)))
    names = ("x", "gain", "q_weight", "q_bias", "k_weight", "k_bias", "v_weight", "v_bias")
    grad_outputs = tuple(torch.randn(BATCH, SEQ_LEN, HIDDEN, device="cuda", dtype=torch.bfloat16)
                         for _ in range(3))

    def project(x, gain, *parameters):
        return _RMSNormQKV.apply(x, gain, buffer, COMPRESS_ACTIVATIONS, act_dist, *parameters)

    # Plain PyTorch reference, before weight compression, with fixed Q/K/V gradients.
    normalized = F.rms_norm(x, [WIDTH], gain.to(x.dtype))
    expected = tuple(F.linear(normalized, layer.weight, layer.bias.to(x.dtype)) for layer in layers)
    torch.autograd.backward(expected, grad_outputs)
    expected = tuple(output.detach() for output in expected)
    expected_grads = [tensor.grad.clone() for tensor in inputs]
    for tensor in inputs:
        tensor.grad = None
    if COMPRESS_WEIGHTS:
        for layer in layers:
            layer.compress_weight(buffer)
    inputs = (x, gain, *(p for layer in layers for p in (layer.weight, layer.bias)))

    run = torch.compile(project) if COMPILE else project
    for step in range(STEPS):
        outputs = run(*inputs)
        torch.autograd.backward(outputs, grad_outputs)
        output_errors = {name: (output.detach().float() - ref.float()).abs().max().item()
                         for name, output, ref in zip(("q", "k", "v"), outputs, expected)}
        errors = {name: (tensor.grad.float() - ref.float()).abs().max().item()
                  for name, tensor, ref in zip(names, inputs, expected_grads)}
        print(f"step {step}: max absolute error: outputs={output_errors}, gradients={errors}", flush=True)
        for tensor in inputs:
            tensor.grad = None


if __name__ == "__main__":
    main()
