# Vendored source provenance

Retrieved 2026-09-18. Only codec/runtime sources and required dependency sources
are included, rather than upstream demos, datasets and model-loading machinery.

| Component | Upstream | Pinned commit | License location |
|---|---|---|---|
| SplitZip | https://github.com/Intelligent-Microsystems-Lab/SplitZip | `caac0872368b344cfab41f1ad84e6569973ad1d5` | `_vendor/splitzip/LICENSE` (MIT) |
| DFloat11 | https://github.com/LeanModels/DFloat11 | `457733886ce6ebc6d8dda1621fad1ffa2661e028` | `_vendor/dfloat11/LICENSE` (Apache-2.0) |
| ZipNN | https://github.com/zipnn/zipnn | `6009704271394f0497dd6352f72aedc6b947bfc0` | `_vendor/zipnn/LICENSE` (MIT) |
| FiniteStateEntropy | https://github.com/Cyan4973/FiniteStateEntropy | `9f30e0918f87bd835fa040d922a208d7b219e50b` | `_vendor/zipnn/include/FiniteStateEntropy/LICENSE` (BSD-2-Clause/GPL-2.0 option; use BSD) |
| dahuffman 0.4.2 | https://github.com/soxofaan/dahuffman | `a7f31950576c0f3538657e12614db69bff9be5c0` | `_vendor/dahuffman/LICENSE` (MIT) |

## Local changes

- SplitZip `codec_gpu.py` is unchanged. Wrapper flattens inputs for upstream's
  dtype reinterpretations and keeps the decode LUT with each encoded tensor.
- DFloat11 `decode.cu` and `decode.ptx` are unchanged. `dfloat11_utils.py` uses
  a relative dahuffman import, removes progress/diagnostic output, and initializes
  the lookup filler from a valid entry before expanding ranges. This fixes an
  uninitialized variable when EOF has the first code (e.g. constant tensors).
  Unused EOF prefixes now map to a valid symbol for decoder lookahead.
  The encoded NumPy byte array is copied to writable storage before conversion
  to PyTorch, avoiding a read-only-buffer warning.
- DFloat11 wrapper adds guard bytes to the encoded/gap allocations, checks the
  decoder's supported exponent range, and launches the original PTX on PyTorch's
  current stream through `ctypes` instead of requiring CuPy. Buffer guard bytes
  count towards retained memory, not logical payload. CPU reference decoding and
  tensor-level wrappers are local additions, not upstream APIs.
- ZipNN Python imports are relative, including the locally built native module.
  `build.py` builds only the copied native sources, with no submodule downloads or
  package installation. C/FSE sources are unchanged. The wrapper protects inputs
  against upstream's in-place bit reorder with a private CPU copy.
- dahuffman exposes only `HuffmanCodec` in its local initializer; bundled text
  codecs are omitted. Its Huffman algorithm is unchanged.

Original copyright headers and license texts are retained alongside the code.
