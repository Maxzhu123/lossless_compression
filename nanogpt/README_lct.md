# LCT training experiment

Run from the repository root:

```bash
python -m nanogpt.train_gpt_lct
```

Edit the options at the top of `train_gpt_lct.py`:

- `COMPRESS_WEIGHTS`: store all matrix weights, including embeddings and the
  vocabulary projection, in LCT between uses.
- `COMPRESS_ACTIVATIONS`: compress saved CUDA BF16 activations with at least
  `MIN_COMPRESS_ELEMENTS` entries. This includes FlashAttention's Q/K/V/output
  saves. FP32 attention statistics, integer indices, and small saves stay dense.
- `COMPRESS_OPTIMISER`: compress Muon momentum only. All AdamW first/second
  moments stay dense regardless of this switch.
- `BUFFER`: use the shared LCT fallback arena sized by `BUFFER_SIZE_MIB`.
  Persistent weights/state and temporary activations share this arena. It is
  never reset between forwards; saved-activation allocations are released when
  autograd releases their owners, including when a graph is retained.

The switches are independent and default to false. Matrices are BF16 in **all**
configurations; biases and RMSNorm gains are FP32. The existing simple trainer
stores its linear matrices in FP32, so use the all-false LCT trainer as the
baseline for compression comparisons. LCT compression itself is lossless.

The architecture retains the simple trainer's rotary embeddings, normalized
Q/K, causal attention scale 0.12, ReLU-squared MLP, logit soft cap, and
AdamW/Muon parameter split and learning-rate schedule. It also retains summed
microbatch losses for gradient accumulation. The transformer uses eager autograd
and native FlashAttention, with method-level compilation for the embedding path
and Muon update. The soft-cap/cross-entropy is compiled as a separate dense-tensor
function, outside custom projection autograd, so validation cannot make the
entire loss fall back to eager execution. Whole-model `torch.compile` is not
enabled. Microbatch size can be tuned without changing tokens per optimizer step.

MLP expansion, attention Q/K/V, and the vocabulary projection use fused
RMSNorm/linear autograd. Forward retains the unnormalized input and reciprocal
RMS statistic, and backward reconstructs the normalized input for weight
gradients. Q/K/V share one norm and saved input. Learned gains, biases, native
RMSNorm epsilon/rounding, and all compression switches are preserved.

FlashAttention uses PyTorch's public SDPA interface with the Flash backend
selected explicitly. Saved-tensor hooks encode/decode the BF16 tensors retained
by its native autograd implementation, avoiding a separate backward kernel or
dependence on private FlashAttention operator signatures. Unpacking does not
free the encoded data, since autograd may request it more than once.
See PyTorch's [saved-tensor hook documentation](https://docs.pytorch.org/docs/stable/notes/autograd)
and [SDPA backend selection](https://docs.pytorch.org/docs/main/generated/torch.nn.attention.sdpa_kernel.html).

FineWeb train/validation shards are read from `nanogpt/data/fineweb10B` using
the existing format. Logs and dense, CPU model checkpoints are written under
`nanogpt/logs/<timestamp>_lct`. Checkpoints preserve parameter names and rotary
buffers and can be loaded into a fresh dense or compressed LCT model using
`load_state_dict`; they do not include optimiser resume state.

For shared layers, call `.to(device)` and initialize/load dense weights before
`compress_weight()`. Compressed leaves are not `nn.Parameter` objects: use
`named_trainable_tensors()` to include them when building optimisers. The GPT
wrapper handles gradient clearing and checkpoint export for both storage modes.

Run the CUDA correctness checks with:

```bash
python -m unittest nanogpt.test_train_gpt_lct -v
```

The checks cover native FlashAttention gradients and retained-graph lifetime,
linear gradients, AdamW parity, all eight switches with/without a buffer,
gradient accumulation, and dense/compressed checkpoint loading.
