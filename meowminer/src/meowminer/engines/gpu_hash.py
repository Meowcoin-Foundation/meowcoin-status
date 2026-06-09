"""Loader for the CUDA hashing extension (csrc/gpu_hash.cu).

Tries, in order: a prebuilt `meowminer_gpu_hash` module (shipped in the rig
bundle by CI), then a JIT build via torch.utils.cpp_extension (needs nvcc on
the rig). Returns None when neither is available — the torch engine then
falls back to CPU blake3, which is correct but ~10-20x slower overall.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("meowminer.gpu_hash")

_CSRC = Path(__file__).resolve().parent.parent.parent.parent / "csrc"
_cached = False
_module = None


def load():
    global _cached, _module
    if _cached:
        return _module
    _cached = True
    try:
        import meowminer_gpu_hash  # prebuilt wheel from CI

        _module = meowminer_gpu_hash
        log.info("using prebuilt meowminer_gpu_hash extension")
        return _module
    except ImportError:
        pass
    try:
        from torch.utils.cpp_extension import load as jit_load

        _module = jit_load(
            name="meowminer_gpu_hash",
            sources=[str(_CSRC / "gpu_hash.cu")],
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
            verbose=False,
        )
        log.info("JIT-built meowminer_gpu_hash extension")
    except Exception as exc:
        log.warning("GPU hash extension unavailable (%s); falling back to CPU blake3", exc)
        _module = None
    return _module


def bound_to_le_words(bound: int):
    """Clamp a (possibly >2^256) bound to U256 and split into 8 LE i32 words."""
    import torch

    bound = min(bound, (1 << 256) - 1)
    words = [(bound >> (32 * i)) & 0xFFFFFFFF for i in range(8)]
    # store as int32 bit patterns
    return torch.tensor([w - (1 << 32) if w >= (1 << 31) else w for w in words], dtype=torch.int32)
