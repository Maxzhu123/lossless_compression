"""Parameter-wise updates for Qwen with optional reentrant checkpointing."""
from LCT.components.sparse_utils import SparseMuon


class OptimizerInBackward:
    def __init__(self, model, optimizers):
        self.optimizers = {}
        self.parameters = []
        for optimizer in optimizers:
            params = optimizer.params if isinstance(optimizer, SparseMuon) else optimizer.param_groups[0]['params']
            assert len(params) == 1
            parameter = params[0]
            self.parameters.append(parameter)
            self.optimizers[id(parameter)] = optimizer
        self.head = model.lm_head.weight
        self.tied_embedding = self.head is model.model.embed_tokens.weight
        self.remaining = {}
        self.handles = [p.register_post_accumulate_grad_hook(self.update) for p in self.parameters]

    def begin(self, head_chunks, checkpoint_head=True):
        self.remaining = {id(p): 1 for p in self.parameters}
        # Each reentrant head chunk accumulates separately. The shared embedding
        # contributes once more, after the transformer backward has finished.
        # Without head checkpointing, autograd combines all contributions in
        # one graph task and fires the shared leaf's hook once.
        self.remaining[id(self.head)] = head_chunks + int(self.tied_embedding) if checkpoint_head else 1

    def update(self, parameter):
        key = id(parameter)
        self.remaining[key] -= 1
        if self.remaining[key] < 0:
            raise RuntimeError('Unexpected extra gradient contribution in optimizer-in-backward')
        if self.remaining[key] == 0:
            self.optimizers[key].step()
            parameter.grad = None

    def finish(self):
        if any(self.remaining.values()) or any(p.grad is not None for p in self.parameters):
            raise RuntimeError('Optimizer-in-backward did not consume every parameter gradient')

    def remove(self):
        for handle in self.handles:
            handle.remove()
