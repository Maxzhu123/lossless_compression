"""CUDA correctness checks: python -m unittest nanogpt.test_train_gpt_lct -v."""
import gc
import itertools
import unittest

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from LCT.LCTensor import MyCompressed
from LCT.layers import Linear, dense_weight, rms_norm_linears
from LCT.saved_tensors import ActivationCompression
from LCT.sparse_utils import SparseAdamW
from LCT.tensor_buffer import TensorBuffer
from nanogpt.train_gpt_lct import GPT, make_optimizers, softcap_cross_entropy


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class LCTTrainingTests(unittest.TestCase):
    def test_loss_validation_then_training(self):
        # Validation is the first caller in train(), unlike test_time_mem.py.
        with torch.no_grad():
            x = torch.randn(4, 8, 256, device='cuda', dtype=torch.bfloat16)
            targets = torch.randint(0, 256, (4, 8), device='cuda')
            self.assertTrue(torch.isfinite(softcap_cross_entropy(x, targets)))
        for batch in (32, 4):
            x = torch.randn(batch, 8, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            reference = x.detach().clone().requires_grad_()
            targets = torch.randint(0, 256, (batch, 8), device='cuda')
            logits = reference.float()
            logits = 15 * logits * (logits.square() + 15**2).rsqrt()
            expected = F.cross_entropy(logits.reshape(-1, 256), targets.flatten(), reduction='sum')
            actual = softcap_cross_entropy(x, targets)
            expected.backward()
            actual.backward()
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(x.grad, reference.grad, rtol=0.01, atol=1e-5)

    def test_fused_rms_norm_linear(self):
        for projections, compressed, activations in itertools.product((1, 3), (False, True), (False, True)):
            with self.subTest(projections=projections, weights=compressed, activations=activations):
                x = torch.randn(2, 17, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
                gain = torch.randn(128, device='cuda', requires_grad=True)
                layers = [Linear(128, 256, bias=(i % 2 == 0)).cuda() for i in range(projections)]
                reference_x = x.detach().clone().requires_grad_()
                reference_gain = gain.detach().clone().requires_grad_()
                reference_parameters = []
                normalized = F.rms_norm(reference_x, (128,), reference_gain.bfloat16())
                expected_outputs = []
                for layer in layers:
                    w = layer.weight.detach().clone().requires_grad_()
                    b = layer.bias.detach().clone().requires_grad_() if layer.bias is not None else None
                    reference_parameters.extend([w] + ([b] if b is not None else []))
                    expected_outputs.append(F.linear(normalized, w, b.bfloat16() if b is not None else None))
                    if compressed:
                        layer.compress_weight()
                grads = [torch.randn_like(y) for y in expected_outputs]
                expected = torch.autograd.grad(expected_outputs, [reference_x, reference_gain, *reference_parameters], grads)
                if activations:
                    with ActivationCompression(min_elements=0):
                        outputs = rms_norm_linears(x, gain, layers)
                else:
                    outputs = rms_norm_linears(x, gain, layers)
                parameters = [p for layer in layers for p in (layer.weight, layer.bias) if p is not None]
                actual = torch.autograd.grad(outputs, [x, gain, *parameters], grads)
                for a, b in zip(outputs, expected_outputs):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_fused_norm_does_not_save_normalized_input(self):
        x = torch.randn(2, 17, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        gain = torch.ones(128, device='cuda', requires_grad=True)
        layers = [Linear(128, 256).cuda() for _ in range(3)]
        saved = []
        def pack(tensor):
            if tensor.shape == x.shape:
                saved.append(tensor.data_ptr())
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            outputs = rms_norm_linears(x, gain, layers)
        self.assertEqual(saved, [x.data_ptr()])
        sum(y.float().sum() for y in outputs).backward()

    def test_flash_attention_gradients_and_buffer_lifetime(self):
        for buffered in (False, True):
            with self.subTest(buffered=buffered):
                buffer = TensorBuffer(8 * 1024**2) if buffered else None
                q, k, v = [torch.randn(2, 65, 2, 128, device='cuda', dtype=torch.bfloat16)
                           .transpose(1, 2).requires_grad_() for _ in range(3)]
                grad = torch.randn_like(q)
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    reference = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=.12)
                expected = torch.autograd.grad(reference, (q, k, v), grad)
                hooks = ActivationCompression(buffer=buffer, min_elements=0)
                with hooks, sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    output = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=.12)
                self.assertIn('FlashAttention', type(output.grad_fn).__name__)
                self.assertEqual(hooks.packed_count, 4)  # Q, K, V and output; not FP32 LSE.
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
                for retain in (True, False):
                    actual = torch.autograd.grad(output, (q, k, v), grad, retain_graph=retain)
                    for a, b in zip(actual, expected):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                del output, hooks
                gc.collect()
                if buffer is not None:
                    torch.cuda.synchronize()
                    self.assertEqual(buffer._free_count.item(), 1)
                    self.assertEqual(buffer._free_sizes[0].item(), buffer.capacity_bytes)

    def test_linear_matches_native(self):
        for compressed in (False, True):
            layer = Linear(128, 256).cuda()
            x = torch.randn(2, 17, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            weight = layer.weight.detach().clone().requires_grad_()
            bias = layer.bias.detach().clone().requires_grad_()
            grad = torch.randn(2, 17, 256, device='cuda', dtype=torch.bfloat16)
            reference = F.linear(x, weight, bias.bfloat16())
            expected = torch.autograd.grad(reference, (x, weight, bias), grad)
            if compressed:
                layer.compress_weight()
            output = layer(x)
            actual = torch.autograd.grad(output, (x, layer.weight, layer.bias), grad)
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_adamw_matches_pytorch(self):
        for dtype, compressed in itertools.product((torch.float32, torch.bfloat16), (False, True)):
            with self.subTest(dtype=dtype, compressed=compressed):
                parameter = torch.randn(128, 128, device='cuda', dtype=dtype, requires_grad=True)
                reference = parameter.detach().clone().requires_grad_()
                ours = SparseAdamW([dict(params=[parameter], lr=.002)], compressed=compressed, weight_decay=.01)
                native = torch.optim.AdamW([reference], lr=.002, betas=(.8, .95), eps=1e-10,
                                          weight_decay=.01, fused=False, foreach=False)
                for _ in range(3):
                    parameter.grad = torch.randn_like(parameter)
                    reference.grad = parameter.grad.clone()
                    ours.step()
                    native.step()
                    torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
                state = ours.state[parameter]
                for key in ('exp_avg', 'exp_avg_sq'):
                    self.assertEqual(isinstance(state[key], MyCompressed), compressed and dtype == torch.bfloat16)

    def test_all_compression_combinations(self):
        torch.manual_seed(4)
        base = GPT(256, 1, 128).cuda()
        checkpoint = base.checkpoint()
        batches = [torch.randint(0, 256, (2, 17), device='cuda') for _ in range(2)]
        for inputs in batches:
            base(inputs, inputs).backward()
        reference_grads = {n: p.grad.clone() for n, p in base.named_trainable_tensors()}
        for weights, activations, optimiser, buffered in itertools.product((False, True), repeat=4):
            with self.subTest(weights=weights, activations=activations, optimiser=optimiser, buffered=buffered):
                model = GPT(256, 1, 128, compress_activations=activations, min_compress_elements=0).cuda()
                model.load_state_dict(checkpoint)
                if buffered:
                    model.set_tensor_buffer(TensorBuffer(32 * 1024**2))
                if weights:
                    model.compress_weights()
                opts = make_optimizers(model, optimiser)
                for inputs in batches:
                    loss = model(inputs, inputs)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                for name, p in model.named_trainable_tensors():
                    torch.testing.assert_close(p.grad, reference_grads[name], rtol=0, atol=0)
                for _ in range(2):
                    for opt in opts:
                        opt.step()
                    model.zero_grad()
                    loss = model(batches[0], batches[0])
                    loss.backward()
                model.zero_grad()
                self.assertTrue(all(p.grad is None for _, p in model.named_trainable_tensors()))
                saved = model.checkpoint()
                clone = GPT(256, 1, 128).cuda()
                clone.load_state_dict(saved)
                model.load_state_dict(saved)  # Loading into compressed leaves too.
                with torch.no_grad():
                    torch.testing.assert_close(model(batches[0], batches[0]), clone(batches[0], batches[0]), rtol=0, atol=0)
                for p in opts[1].momentums:
                    self.assertEqual(isinstance(p, MyCompressed), optimiser)
                for state in opts[0].state.values():
                    for key in ('exp_avg', 'exp_avg_sq'):
                        self.assertNotIsInstance(state[key], MyCompressed)
                del model, clone, opts, loss
                gc.collect()


if __name__ == '__main__':
    unittest.main()
