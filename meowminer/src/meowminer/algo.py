"""Pearlhash algorithm primitives, mirrored from pearl-research-labs/pearl.

Sources of truth (file references are into the Pearl monorepo):
  * job key / commitment seeds: zk-pow/src/ffi/mine.rs::{compute_job_key,
    compute_commitment_hash}
  * jackpot accumulation:        zk-pow/src/ffi/mine.rs::try_mine_one and
    zk-pow/src/circuit/pearl_program.rs (JACKPOT_SIZE=16, LROT_PER_TILE=13)
  * jackpot hash:                zk-pow/src/api/proof_utils.rs::compute_jackpot_hash
  * periodic patterns:           zk-pow/src/api/proof_utils.rs (PeriodicPattern)

Everything here is CPU/numpy and serves two roles: the reference engine (slow
but correct, used for tests and share validation before submit) and the shared
plumbing for the GPU engine (which reproduces the same numbers with tensor-core
GEMMs).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from blake3 import blake3

JACKPOT_SIZE = 16
LROT_PER_TILE = 13
BLAKE3_CHUNK_LEN = 1024
SIGNAL_MIN, SIGNAL_MAX = -64, 64


# ── periodic patterns ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PeriodicPattern:
    """Generalized arithmetic progression: {a*s0 + b*s1 + c*s2}."""

    shape: tuple[tuple[int, int], ...]  # 3 x (stride, length)

    def to_list(self) -> list[int]:
        res = [0]
        for stride, length in self.shape:
            res = [r + i * stride for i in range(length) for r in res]
        return res

    @property
    def period(self) -> int:
        stride, length = self.shape[-1]
        return stride * length

    @property
    def size(self) -> int:
        out = 1
        for _, length in self.shape:
            out *= length
        return out

    def offset_is_valid(self, offset: int) -> bool:
        for stride, length in reversed(self.shape):
            offset %= stride * length
            if offset >= stride:
                return False
        return True

    def partitions(self, total_dimension: int) -> list[list[int]]:
        """All index groups for one matrix dimension (mine.rs::threads_partition)."""
        if total_dimension % self.period != 0:
            raise ValueError("total_dimension must be divisible by pattern period")
        base = self.to_list()
        return [
            [offset + d for d in base]
            for offset in range(total_dimension)
            if self.offset_is_valid(offset)
        ]

    def valid_offsets(self, total_dimension: int) -> list[int]:
        return [o for o in range(total_dimension) if self.offset_is_valid(o)]


@dataclass(frozen=True)
class MiningConfig:
    """Python mirror of pearl_mining.MiningConfiguration for the hot path.

    ``to_bytes`` must produce the exact 52-byte serialization used in
    job_key = blake3(header || config); we delegate that to the PyO3 binding at
    runtime (see proofs.py) and carry the raw bytes here so the hot path has no
    native dependency.
    """

    common_dim: int  # k
    rank: int
    rows_pattern: PeriodicPattern
    cols_pattern: PeriodicPattern
    config_bytes: bytes  # exact 52-byte serialization from pearl_mining

    @property
    def tile_h(self) -> int:
        return self.rows_pattern.size

    @property
    def tile_w(self) -> int:
        return self.cols_pattern.size

    @property
    def dot_product_length(self) -> int:
        return self.common_dim - self.common_dim % self.rank

    @property
    def difficulty_adjustment(self) -> int:
        return self.tile_h * self.tile_w * self.dot_product_length


# ── hashing / seeds ──────────────────────────────────────────────────────────


def pad_to_chunk_boundary(data: bytes) -> bytes:
    rem = len(data) % BLAKE3_CHUNK_LEN
    return data if rem == 0 else data + b"\x00" * (BLAKE3_CHUNK_LEN - rem)


def compute_job_key(header_bytes: bytes, config_bytes: bytes) -> bytes:
    return blake3(header_bytes + config_bytes).digest()


def compute_commitment_seeds(
    job_key: bytes, a_row_major: bytes, b_col_major: bytes
) -> tuple[bytes, bytes]:
    """Return (b_noise_seed, a_noise_seed) — mine.rs::compute_commitment_hash.

    Inputs must already be chunk-padded; the hashes double as the matrices'
    Merkle roots, which is what binds the noise to the committed matrices.
    """
    hash_a = blake3(a_row_major, key=job_key).digest()
    hash_b = blake3(b_col_major, key=job_key).digest()
    b_noise_seed = blake3(job_key + hash_b).digest()
    a_noise_seed = blake3(b_noise_seed + hash_a).digest()
    return b_noise_seed, a_noise_seed


def compute_jackpot_hash(jackpot: np.ndarray, a_noise_seed: bytes) -> bytes:
    """blake3 of the 16xu32 jackpot (LE bytes), keyed with a_noise_seed."""
    assert jackpot.dtype == np.uint32 and jackpot.size == JACKPOT_SIZE
    return blake3(jackpot.astype("<u4").tobytes(), key=a_noise_seed).digest()


# ── noise ────────────────────────────────────────────────────────────────────


def dense_noise(a_noise_seed: bytes, b_noise_seed: bytes, m: int, k: int, n: int, rank: int):
    """Dense (m x k) A-noise and (k x n) B-noise via the low-rank factors.

    Wraps the vendored reference NoiseGenerator (bit-identical to pearl's
    miner-base). Returns int16 numpy arrays (entries fit in [-63, 63]).
    """
    import torch

    from meowminer.vendor.noise_generation import NoiseGenerator

    gen = NoiseGenerator(noise_rank=rank, noise_range=128)
    a_l, a_r, b_l, b_r = gen.generate_noise_metrices(a_noise_seed, b_noise_seed, m, k, n)
    noise_a = (a_l.to(torch.int32) @ a_r.to(torch.int32)).numpy()
    noise_b = (b_l.to(torch.int32) @ b_r.to(torch.int32)).numpy()
    return noise_a.astype(np.int16), noise_b.astype(np.int16)


# ── jackpot accumulation (reference) ─────────────────────────────────────────


def jackpots_for_attempt(
    a_noised: np.ndarray,  # (m, k) int32
    bt_noised: np.ndarray,  # (n, k) int32  (B^T rows = B cols)
    config: MiningConfig,
) -> tuple[np.ndarray, list[tuple[list[int], list[int]]]]:
    """Compute the 16xu32 jackpot for every (a_rows, b_cols) partition pair.

    Returns (jackpots, partitions) where jackpots has shape
    (num_row_parts * num_col_parts, 16) uint32 and partitions[i] is the
    (a_rows, b_cols) index lists that produced jackpots[i].

    Vectorized restatement of the scalar loop in mine.rs::try_mine_one: for
    each rank-block ``ll`` the *cumulative* tile C_cum = A_sel @ B_sel^T over
    columns [0, ll) is XOR-folded to one u32 and rotl-mixed into
    jackpot[(ll/rank - 1) % 16].
    """
    m, k = a_noised.shape
    n, _ = bt_noised.shape
    rank = config.rank
    row_parts = config.rows_pattern.partitions(m)
    col_parts = config.cols_pattern.partitions(n)
    h, w = config.tile_h, config.tile_w
    num_blocks = config.dot_product_length // rank

    # Gather: (R, h, k) and (C, w, k)
    a_sel = a_noised[np.asarray(row_parts)]  # (R, h, k)
    b_sel = bt_noised[np.asarray(col_parts)]  # (C, w, k)

    jackpots = np.zeros((len(row_parts), len(col_parts), JACKPOT_SIZE), dtype=np.uint32)
    # cumulative tiles: (R, C, h, w) int32 with wrapping arithmetic
    cum = np.zeros((len(row_parts), len(col_parts), h, w), dtype=np.int64)

    for blk in range(num_blocks):
        lo, hi = blk * rank, (blk + 1) * rank
        # (R, h, rank) x (C, w, rank) -> (R, C, h, w)
        part = np.einsum(
            "rhx,cwx->rchw",
            a_sel[:, :, lo:hi].astype(np.int64),
            b_sel[:, :, lo:hi].astype(np.int64),
        )
        cum += part
        tiles_u32 = (cum & 0xFFFFFFFF).astype(np.uint32)  # i32 wrap -> u32 view
        xored = np.bitwise_xor.reduce(tiles_u32.reshape(len(row_parts), len(col_parts), -1), axis=2)
        tid = blk % JACKPOT_SIZE
        j = jackpots[:, :, tid]
        jackpots[:, :, tid] = ((j << LROT_PER_TILE) | (j >> (32 - LROT_PER_TILE))) ^ xored

    parts = [(r, c) for r in row_parts for c in col_parts]
    return jackpots.reshape(-1, JACKPOT_SIZE), parts


def grade_jackpots(
    jackpots: np.ndarray, a_noise_seed: bytes, bound: int
) -> list[tuple[int, bytes]]:
    """Hash every jackpot and return [(index, hash_jackpot)] entries <= bound."""
    hits = []
    for i in range(jackpots.shape[0]):
        h = compute_jackpot_hash(jackpots[i], a_noise_seed)
        if int.from_bytes(h, "little") <= bound:
            hits.append((i, h))
    return hits
