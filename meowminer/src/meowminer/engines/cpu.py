"""Reference CPU engine: bit-exact, slow, used for tests and sanity checks."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from ..algo import (
    MiningConfig,
    SIGNAL_MAX,
    SIGNAL_MIN,
    compute_commitment_seeds,
    compute_job_key,
    dense_noise,
    grade_jackpots,
    jackpots_for_attempt,
    pad_to_chunk_boundary,
)
from ..job import Job
from .base import AttemptStats, Hit


class CpuEngine:
    name = "cpu-reference"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._config: MiningConfig | None = None
        self._m = self._n = 0

    def configure(self, config: MiningConfig, m: int, n: int) -> None:
        if m % config.rows_pattern.period or n % config.cols_pattern.period:
            raise ValueError("m / n must be divisible by the pattern periods")
        if config.common_dim % config.rank:
            raise ValueError("common_dim must be a multiple of rank")
        self._config, self._m, self._n = config, m, n

    def mine(self, job: Job, share_bound: int, block_bound: int) -> Iterator[tuple[list[Hit], AttemptStats]]:
        assert self._config is not None, "configure() first"
        cfg, m, n, k = self._config, self._m, self._n, self._config.common_dim
        job_key = compute_job_key(job.header, cfg.config_bytes)

        while True:
            a = self._rng.integers(SIGNAL_MIN, SIGNAL_MAX + 1, size=(m, k), dtype=np.int8)
            bt = self._rng.integers(SIGNAL_MIN, SIGNAL_MAX + 1, size=(n, k), dtype=np.int8)

            a_bytes = pad_to_chunk_boundary(a.astype(np.uint8).tobytes())
            bt_bytes = pad_to_chunk_boundary(bt.astype(np.uint8).tobytes())
            b_seed, a_seed = compute_commitment_seeds(job_key, a_bytes, bt_bytes)

            noise_a, noise_b = dense_noise(a_seed, b_seed, m, k, n, cfg.rank)
            a_noised = a.astype(np.int32) + noise_a.astype(np.int32)
            bt_noised = bt.astype(np.int32) + noise_b.T.astype(np.int32)

            jackpots, parts = jackpots_for_attempt(a_noised, bt_noised, cfg)
            hits = [
                Hit(
                    job=job,
                    a_matrix=a,
                    bt_matrix=bt,
                    a_rows=parts[i][0],
                    b_cols=parts[i][1],
                    hash_jackpot=h,
                    is_block=int.from_bytes(h, "little") <= block_bound,
                )
                for i, h in grade_jackpots(jackpots, a_seed, share_bound)
            ]
            stats = AttemptStats(
                attempts=1,
                partitions=len(parts),
                macs=len(parts) * cfg.difficulty_adjustment,
            )
            yield hits, stats
