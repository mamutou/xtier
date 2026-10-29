# tests

Unit tests: kernel numerics against the reference, expert placement, multi-GPU emulation.

No checkpoint and no GPU needed (any ISA, so also the acceptance test for a new one):

- `neon_check.py` — one synthetic MoE layer against a float64 reference (both FP4 formats, 1 to 2048 tokens)
  plus per-token-count timings.
- `real_layer_check.py` — one real checkpoint layer (any MXFP4/NVFP4 checkpoint, no GPU, no SGLang): numerics
  against a float64 reference, timings, and `--save` / `--compare` for two kernel builds.

The rest read a real layer from a checkpoint path inside `layers.py` (`engine_ab.py`, `mailbox_check.py`, ...).
