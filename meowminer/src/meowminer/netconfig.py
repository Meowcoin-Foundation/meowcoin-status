"""MiningConfiguration resolution.

Key insight (verified against zk-pow): the pool/chain verifier RECONSTRUCTS the
MiningConfiguration from the submitted plain proof itself — patterns via
``list_to_pattern(row_indices)``, ``k``/``rank`` from the proof fields
(zk-pow/src/ffi/plain_proof.rs::parse_plain_proof). The miner therefore picks
its own dimensions, subject to the verifier's sanity checks:

  * k >= 16 * rank             ("k must be >= 16r")
  * tile h*w >= 32             ("Inner hash size must be >= 32")
  * rank: power of two, multiple of 32 (blake3 digest size)
  * signal entries in [-64, 64]

Share economics are dimension-neutral (the h*w*k difficulty adjustment exactly
offsets the per-partition work), so the choice is purely a throughput tuning
knob. Defaults below mirror the reference vLLM miner's hash-tile pattern
(miner-base/src/miner_base/settings.py) with a k sized for standalone mining.
"""

from __future__ import annotations

from .algo import MiningConfig, PeriodicPattern

# Reference miner hash-tile pattern (miner-base settings.py): 2 x 64 = 128 elems.
DEFAULT_ROWS = [0, 8]
DEFAULT_COLS = [
    0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57,
    64, 65, 72, 73, 80, 81, 88, 89, 96, 97, 104, 105, 112, 113, 120, 121,
    128, 129, 136, 137, 144, 145, 152, 153, 160, 161, 168, 169, 176, 177, 184, 185,
    192, 193, 200, 201, 208, 209, 216, 217, 224, 225, 232, 233, 240, 241, 248, 249,
]
DEFAULT_RANK = 128
DEFAULT_COMMON_DIM = 4096  # >= 16 * rank; bigger k amortizes per-attempt overhead


def _pattern_from_pm(p) -> PeriodicPattern:
    return PeriodicPattern(shape=tuple((int(s), int(l)) for s, l in p.shape))


def load_config(config_bytes_hex: str | None = None) -> MiningConfig:
    import pearl_mining as pm

    if config_bytes_hex:
        cfg = pm.MiningConfiguration.from_bytes(bytes.fromhex(config_bytes_hex))
    else:
        cfg = pm.MiningConfiguration(
            DEFAULT_COMMON_DIM,
            DEFAULT_RANK,
            pm.MMAType.Int7xInt7ToInt32,
            pm.PeriodicPattern.from_list(DEFAULT_ROWS),
            pm.PeriodicPattern.from_list(DEFAULT_COLS),
            bytes(pm.MiningConfiguration.RESERVED),
        )
    return MiningConfig(
        common_dim=int(cfg.common_dim),
        rank=int(cfg.rank),
        rows_pattern=_pattern_from_pm(cfg.rows_pattern),
        cols_pattern=_pattern_from_pm(cfg.cols_pattern),
        config_bytes=bytes(cfg.to_bytes()),
    )
