"""Algorithm primitives vs a direct transcription of zk-pow's scalar reference.

The scalar implementation below is a line-by-line port of
zk-pow/src/ffi/mine.rs::try_mine_one's jackpot loop; the vectorized
meowminer.algo.jackpots_for_attempt must match it bit-for-bit.
"""

import sys
from pathlib import Path

import numpy as np
from blake3 import blake3

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meowminer.algo import (
    JACKPOT_SIZE,
    LROT_PER_TILE,
    MiningConfig,
    PeriodicPattern,
    compute_commitment_seeds,
    compute_jackpot_hash,
    compute_job_key,
    jackpots_for_attempt,
    pad_to_chunk_boundary,
)


def scalar_jackpot(a_noised, bt_noised, a_rows, b_cols, rank, k):
    """Direct port of the rust loop (i32 wrapping arithmetic)."""
    h, w = len(a_rows), len(b_cols)
    tile = [[0] * w for _ in range(h)]
    jackpot = [0] * JACKPOT_SIZE
    for ll in range(rank, k + 1, rank):
        for u, ai in enumerate(a_rows):
            for v, bi in enumerate(b_cols):
                acc = tile[u][v]
                for l in range(ll - rank, ll):
                    acc = (acc + int(a_noised[ai][l]) * int(bt_noised[bi][l])) & 0xFFFFFFFF
                tile[u][v] = acc
        xored = 0
        for row in tile:
            for x in row:
                xored ^= x
        tid = (ll // rank - 1) % JACKPOT_SIZE
        j = jackpot[tid]
        jackpot[tid] = (((j << LROT_PER_TILE) | (j >> (32 - LROT_PER_TILE))) & 0xFFFFFFFF) ^ xored
    return jackpot


def small_config():
    # tiny: tile 2x2, rank 32, k 64. Canonical shape for the index list {0,1}
    # (PeriodicPattern::from_list pads with (period, 1) tuples, period = 2).
    rows = PeriodicPattern(shape=((1, 2), (2, 1), (2, 1)))
    cols = PeriodicPattern(shape=((1, 2), (2, 1), (2, 1)))
    return MiningConfig(common_dim=64, rank=32, rows_pattern=rows, cols_pattern=cols, config_bytes=b"\x00" * 52)


def test_pattern_helpers():
    p = PeriodicPattern(shape=((1, 2), (2, 1), (2, 1)))
    assert p.to_list() == [0, 1]
    assert p.period == 2
    assert p.size == 2
    # consecutive-pair tiles exactly cover the dimension
    assert p.partitions(8) == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert p.valid_offsets(8) == [0, 2, 4, 6]


def test_vectorized_jackpots_match_scalar_reference():
    rng = np.random.default_rng(7)
    cfg = small_config()
    m = n = 8
    a_noised = rng.integers(-127, 128, size=(m, cfg.common_dim), dtype=np.int32)
    bt_noised = rng.integers(-127, 128, size=(n, cfg.common_dim), dtype=np.int32)

    jackpots, parts = jackpots_for_attempt(a_noised, bt_noised, cfg)
    assert len(parts) == 16  # 4 row partitions x 4 col partitions
    for i, (a_rows, b_cols) in enumerate(parts):
        expected = scalar_jackpot(a_noised, bt_noised, a_rows, b_cols, cfg.rank, cfg.common_dim)
        assert jackpots[i].tolist() == expected, f"partition {i} ({a_rows}x{b_cols})"


def test_jackpot_hash_is_keyed_blake3_of_le_words():
    j = np.arange(16, dtype=np.uint32)
    seed = bytes(range(32))
    expected = blake3(b"".join(int(x).to_bytes(4, "little") for x in j), key=seed).digest()
    assert compute_jackpot_hash(j, seed) == expected


def test_commitment_seed_chain():
    job_key = blake3(b"header" + b"config").digest()
    a = pad_to_chunk_boundary(b"\x01" * 100)
    b = pad_to_chunk_boundary(b"\x02" * 100)
    b_seed, a_seed = compute_commitment_seeds(job_key, a, b)
    hash_a = blake3(a, key=job_key).digest()
    hash_b = blake3(b, key=job_key).digest()
    assert b_seed == blake3(job_key + hash_b).digest()
    assert a_seed == blake3(b_seed + hash_a).digest()


def test_pad_to_chunk_boundary():
    assert len(pad_to_chunk_boundary(b"x" * 1024)) == 1024
    assert len(pad_to_chunk_boundary(b"x" * 1025)) == 2048
    assert pad_to_chunk_boundary(b"") == b""


def test_job_key():
    assert compute_job_key(b"h", b"c") == blake3(b"hc").digest()
