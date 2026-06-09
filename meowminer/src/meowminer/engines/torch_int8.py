"""GPU engine v0: tensor-core int8 GEMM via torch, GPU jackpot folding.

What runs where:
  * signal generation, noising, slice GEMMs (torch._int_mm -> cuBLASLt IMMA),
    cumulative-tile XOR folding and rotl mixing: GPU
  * matrix commitment (keyed blake3 over chunk-padded A/B bytes) and the
    per-partition 64-byte jackpot hashes: CPU (the `blake3` wheel, rayon
    multithreaded)

The CPU hashing is the known bottleneck of v0 (see HANDOFF.md): the design
keeps a pluggable ``hasher`` seam so the phase-2 CUDA extension (vendored
pearl-gemm tensor_hash kernels built for sm_89) drops in without touching the
GEMM loop. Expect single-digit TH/s from v0; the kernel work is what takes it
past SRBMiner.

Correctness is the non-negotiable part: every step mirrors
zk-pow/src/ffi/mine.rs and is cross-checked against the numpy reference in
tests/test_cross_engine.py.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

import numpy as np
from blake3 import blake3

from ..algo import (
    JACKPOT_SIZE,
    LROT_PER_TILE,
    MiningConfig,
    SIGNAL_MAX,
    SIGNAL_MIN,
    compute_commitment_seeds,
    compute_job_key,
    pad_to_chunk_boundary,
)
from ..job import Job
from .base import AttemptStats, Hit

log = logging.getLogger("meowminer.engine.torch")


class TorchInt8Engine:
    name = "torch-int8"

    def __init__(self, device: str = "cuda", m: int = 4096, n: int = 4096, hash_workers: int = 8) -> None:
        import torch  # deferred so CPU-only installs can import the package

        self.torch = torch
        self.device = torch.device(device)
        self._m, self._n = m, n
        self._cfg: MiningConfig | None = None
        self._pool = ThreadPoolExecutor(max_workers=hash_workers, thread_name_prefix="jackpot-hash")

    # -- setup ---------------------------------------------------------------

    def configure(self, config: MiningConfig, m: int | None = None, n: int | None = None) -> None:
        torch = self.torch
        if m:
            self._m = m
        if n:
            self._n = n
        if self._m % config.rows_pattern.period or self._n % config.cols_pattern.period:
            raise ValueError("m / n must be divisible by the pattern periods")
        if config.common_dim % config.rank:
            raise ValueError("common_dim must be a multiple of rank")
        self._cfg = config

        row_parts = config.rows_pattern.partitions(self._m)
        col_parts = config.cols_pattern.partitions(self._n)
        self._row_parts, self._col_parts = row_parts, col_parts
        # Stacked gather indices: selected rows laid out partition-major so the
        # GEMM output is directly viewable as (R, h, C, w) tiles.
        self._row_idx = torch.tensor([i for p in row_parts for i in p], device=self.device, dtype=torch.long)
        self._col_idx = torch.tensor([i for p in col_parts for i in p], device=self.device, dtype=torch.long)
        log.info(
            "configured %s: m=%d n=%d k=%d rank=%d tiles=%dx%d partitions=%d",
            self.name, self._m, self._n, config.common_dim, config.rank,
            config.tile_h, config.tile_w, len(row_parts) * len(col_parts),
        )

    # -- hot loop ------------------------------------------------------------

    def mine(self, job: Job, share_bound: int, block_bound: int) -> Iterator[tuple[list[Hit], AttemptStats]]:
        assert self._cfg is not None, "configure() first"
        torch, cfg = self.torch, self._cfg
        m, n, k, rank = self._m, self._n, cfg.common_dim, cfg.rank
        h, w = cfg.tile_h, cfg.tile_w
        R, C = len(self._row_parts), len(self._col_parts)
        num_blocks = cfg.dot_product_length // rank
        job_key = compute_job_key(job.header, cfg.config_bytes)

        gen = torch.Generator(device=self.device)

        while True:
            # 1. draw signal int7-safe in [-64, 64] (reference range; +noise fits int8)
            a8 = torch.randint(SIGNAL_MIN, SIGNAL_MAX + 1, (m, k), dtype=torch.int8, device=self.device, generator=gen)
            bt8 = torch.randint(SIGNAL_MIN, SIGNAL_MAX + 1, (n, k), dtype=torch.int8, device=self.device, generator=gen)

            # 2. commitment seeds (CPU blake3 over the exact committed bytes)
            a_np = a8.cpu().numpy()
            bt_np = bt8.cpu().numpy()
            a_bytes = pad_to_chunk_boundary(a_np.astype(np.uint8).tobytes())
            bt_bytes = pad_to_chunk_boundary(bt_np.astype(np.uint8).tobytes())
            b_seed, a_seed = compute_commitment_seeds(job_key, a_bytes, bt_bytes)

            # 3. deterministic noise (reference generator), applied on GPU
            from ..vendor.noise_generation import NoiseGenerator

            ng = NoiseGenerator(noise_rank=rank, noise_range=128)
            a_l, a_r, b_l, b_r = ng.generate_noise_metrices(a_seed, b_seed, m, k, n)
            noise_a = (a_l.to(self.device, torch.int32) @ a_r.to(self.device, torch.int32))
            noise_bt = (b_l.to(self.device, torch.int32) @ b_r.to(self.device, torch.int32)).T.contiguous()
            a_noised = (a8.to(torch.int32) + noise_a).to(torch.int8)
            bt_noised = (bt8.to(torch.int32) + noise_bt).to(torch.int8)

            # 4. gather partition-major stacked operands
            a_sel = a_noised.index_select(0, self._row_idx)  # (R*h, k)
            bt_sel = bt_noised.index_select(0, self._col_idx)  # (C*w, k)

            # 5. rank-block cumulative GEMM + XOR/rotl folding, all on GPU
            jack = torch.zeros(R, C, JACKPOT_SIZE, dtype=torch.int32, device=self.device)
            cum = torch.zeros(R * h, C * w, dtype=torch.int32, device=self.device)
            for blk in range(num_blocks):
                lo, hi = blk * rank, (blk + 1) * rank
                cum += torch._int_mm(a_sel[:, lo:hi].contiguous(), bt_sel[:, lo:hi].T.contiguous())
                tiles = cum.view(R, h, C, w)
                xored = tiles.permute(0, 2, 1, 3).reshape(R, C, h * w)
                xored = _xor_reduce_last(xored)
                tid = blk % JACKPOT_SIZE
                j = jack[:, :, tid]
                jack[:, :, tid] = _rotl32(j, LROT_PER_TILE) ^ xored

            # 6. grade: per-partition keyed blake3 of the 64-byte jackpot (CPU pool)
            jack_np = jack.cpu().numpy().astype("<i4").view("<u4").reshape(R * C, JACKPOT_SIZE)
            hits = self._grade(job, a_np, bt_np, jack_np, a_seed, share_bound, block_bound)

            stats = AttemptStats(attempts=1, partitions=R * C, macs=R * C * cfg.difficulty_adjustment)
            yield hits, stats

    # -- grading -------------------------------------------------------------

    def _grade(self, job, a_np, bt_np, jack_np, a_seed, share_bound, block_bound) -> list[Hit]:
        def hash_one(i: int) -> tuple[int, bytes]:
            return i, blake3(jack_np[i].tobytes(), key=a_seed).digest()

        hits = []
        for i, digest in self._pool.map(hash_one, range(jack_np.shape[0]), chunksize=2048):
            if int.from_bytes(digest, "little") <= share_bound:
                r, c = divmod(i, len(self._col_parts))
                hits.append(
                    Hit(
                        job=job,
                        a_matrix=a_np,
                        bt_matrix=bt_np,
                        a_rows=self._row_parts[r],
                        b_cols=self._col_parts[c],
                        hash_jackpot=digest,
                        is_block=int.from_bytes(digest, "little") <= block_bound,
                    )
                )
        return hits


def _xor_reduce_last(t):
    """XOR-reduce the last dimension (zero-pads to a power of two; 0 is the XOR identity)."""
    size = t.shape[-1]
    if size & (size - 1):
        import torch

        pow2 = 1 << size.bit_length()
        pad = torch.zeros(*t.shape[:-1], pow2 - size, dtype=t.dtype, device=t.device)
        t = torch.cat([t, pad], dim=-1)
    while t.shape[-1] > 1:
        half = t.shape[-1] // 2
        t = t[..., :half] ^ t[..., half:]
    return t[..., 0]


def _rotl32(t, bits: int):
    """Logical rotl on int32 tensors (torch >> is arithmetic; mask the fill)."""
    left = t << bits
    right = (t >> (32 - bits)) & ((1 << bits) - 1)
    return left | right
