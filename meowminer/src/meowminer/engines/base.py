"""Engine interface: turn a Job into share hits as fast as the silicon allows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np

from ..algo import MiningConfig
from ..job import Job


@dataclass
class Hit:
    """One candidate that cleared the share bound.

    Carries everything proofs.build_plain_proof needs: the *unnoised* matrices
    (the proof commits to them), the winning partition's row/col indices, and
    the jackpot hash for logging / block-bound checking.
    """

    job: Job
    a_matrix: np.ndarray  # (m, k) int8, raw signal
    bt_matrix: np.ndarray  # (n, k) int8, raw B^T
    a_rows: list[int]
    b_cols: list[int]
    hash_jackpot: bytes
    is_block: bool = False


@dataclass
class AttemptStats:
    attempts: int = 0  # (A,B) pairs tried
    partitions: int = 0  # jackpot candidates graded
    macs: int = 0  # multiply-accumulates executed (hashrate basis)


class Engine(Protocol):
    name: str

    def configure(self, config: MiningConfig, m: int, n: int) -> None: ...

    def mine(self, job: Job, share_bound: int, block_bound: int) -> Iterator[tuple[list[Hit], AttemptStats]]:
        """Yield (hits, stats) batches until the caller stops iterating.

        Implementations must re-check ``job`` freshness only via the caller
        (the orchestrator swaps jobs by tearing down the iterator), keeping
        engines free of synchronization concerns.
        """
        ...
