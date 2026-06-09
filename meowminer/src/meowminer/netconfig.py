"""Network MiningConfiguration resolution.

The pool's mining.notify carries only the header; the MiningConfiguration
(common_dim k, rank, row/col patterns) is a network constant the miner must
already know — it feeds job_key, the noise, and the difficulty adjustment, so
a mismatch means 100% rejects.

Resolution order:
  1. explicit --config-bytes hex on the CLI (52 bytes)
  2. pearl_mining's default MiningConfiguration (tracks the deployed network)

The desktop session should confirm the live mainnet values once against a
running pearld (getblocktemplate carries them) and pin them in mfarm's flight
sheet; see HANDOFF.md.
"""

from __future__ import annotations

from .algo import MiningConfig, PeriodicPattern


def _pattern_from_pm(p) -> PeriodicPattern:
    shape = tuple((int(s), int(l)) for s, l in p.shape)
    return PeriodicPattern(shape=shape)


def load_config(config_bytes_hex: str | None = None) -> MiningConfig:
    import pearl_mining as pm

    if config_bytes_hex:
        raw = bytes.fromhex(config_bytes_hex)
        cfg = pm.MiningConfiguration.from_bytes(raw)
    else:
        cfg = pm.MiningConfiguration.default()
        raw = bytes(cfg.to_bytes())
    return MiningConfig(
        common_dim=int(cfg.common_dim),
        rank=int(cfg.rank),
        rows_pattern=_pattern_from_pm(cfg.rows_pattern),
        cols_pattern=_pattern_from_pm(cfg.cols_pattern),
        config_bytes=raw,
    )
