import torch
from cprint import c_print

from LCT.tensor_buffer import TensorBuffer
from sparse_utils import SparseSGDM
from mlp_train import Model

COMPRESSED = True
BUFFER = False
c_print(f"Compressed: {COMPRESSED}", color="blue")
c_print(f"Buffer: {BUFFER}", color="blue")


def inspect_saved_tensors(module, *inputs, **kwinputs):
    saved = []
    current_module = [None]
    handles = []

    for name, submodule in module.named_modules():
        display_name = f"{name}.{submodule.__class__.__name__}" if name else submodule.__class__.__name__

        def make_pre_hook(display_name):
            def pre_hook(module, inputs):
                current_module[0] = display_name
            return pre_hook

        def post_hook(module, inputs, output):
            current_module[0] = None

        handles.append(submodule.register_forward_pre_hook(make_pre_hook(display_name)))
        handles.append(submodule.register_forward_hook(post_hook))

    def pack_hook(tensor):
        saved.append((current_module[0], tensor))
        return tensor

    try:
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda x: x):
            output = module(*inputs, **kwinputs)
    finally:
        for handle in handles:
            handle.remove()

    return output, saved


def main():
    G = torch.Generator(device="cuda")
    G.manual_seed(0)

    if BUFFER and COMPRESSED:
        buffer = TensorBuffer(500_000_000)
    else:
        buffer = None

    model = Model(8, 4096, 21504, 4096, G, buffer=buffer)
    optimiser = SparseSGDM(model.sparse_parameters(), lr=0.001, momentum=0.9,
                           buffer=buffer, compressed=COMPRESSED)

    x = torch.randn(12000, 4096, dtype=torch.bfloat16, device="cuda", generator=G)
    y_hat = x.norm(dim=0)

    # Warmup
    for i in range(2):
        y = model(x, buffer=buffer)
        loss = (y - y_hat).pow(2).mean()
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

    # Main run
    y, saved = inspect_saved_tensors(model, x, buffer=buffer)

    for name, t in saved:

        print(f"Saved tensor: {name}, layout: {t.layout}, shape: {t.shape}")


if __name__ == "__main__":
    main()
