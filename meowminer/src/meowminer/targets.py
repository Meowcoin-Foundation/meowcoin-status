"""Target / difficulty math for Pearlhash pool mining.

Mirrors the consensus math in the Pearl repo (zk-pow) and P2Pearl exactly:

* ``nbits_to_difficulty``  — zk-pow/src/api/proof_utils.rs::nbits_to_difficulty
* ``target_to_bits``       — P2Pearl/src/p2pearl/consensus/difficulty.py::target_to_bits
* ``share_bound``          — zk-pow/src/api/sanity_checks.rs::extract_difficulty_bound

The subtlety that costs you shares if you get it wrong: pools grade a submitted
plain proof with ``verify_plain_proof_with_nbits(header, proof, share_nbits)``.
The Rust verifier decodes the compact nbits and then multiplies by the
difficulty-adjustment factor ``h * w * k`` (tile height x tile width x common
dim).  A miner must therefore grade candidates locally against

    bound = nbits_to_difficulty(target_to_bits(share_target)) * h * w * k

NOT against the raw 256-bit ``target`` field from ``mining.notify`` — grading at
the raw target is ~2**19 times too strict and silently discards almost every
share the pool would have paid you for.

``hash_jackpot`` is compared as a LITTLE-endian 256-bit integer.
"""

from __future__ import annotations

U256_MAX = (1 << 256) - 1


def nbits_to_difficulty(nbits: int) -> int:
    """Decode Bitcoin-compact nbits to a 256-bit integer target.

    Returns 0 for invalid encodings (zero mantissa/exponent or sign bit set),
    matching the Rust implementation.
    """
    exponent = (nbits >> 24) & 0xFF
    mantissa = nbits & 0x00FFFFFF
    if mantissa == 0 or exponent == 0:
        return 0
    if mantissa & 0x00800000:
        return 0  # sign bit set -> invalid
    if exponent <= 3:
        target = mantissa >> (8 * (3 - exponent))
    else:
        target = mantissa << (8 * (exponent - 3))
    return target & U256_MAX


def target_to_bits(target: int) -> int:
    """Encode a 256-bit integer target as Bitcoin-compact nbits."""
    if target <= 0:
        return 0
    nbytes = (target.bit_length() + 7) // 8
    if nbytes <= 3:
        mantissa = target << (8 * (3 - nbytes))
    else:
        mantissa = target >> (8 * (nbytes - 3))
    if mantissa & 0x00800000:
        mantissa >>= 8
        nbytes += 1
    return (nbytes << 24) | (mantissa & 0x007FFFFF)


def share_bound(share_target: int, tile_h: int, tile_w: int, common_dim: int) -> int:
    """The bound the pool will actually grade our shares at.

    Round-trips the 256-bit share target through compact nbits (losing the same
    precision the pool's verifier loses) and applies the h*w*k adjustment, so
    local grading is bit-identical to pool-side acceptance.
    """
    nbits = target_to_bits(share_target)
    return nbits_to_difficulty(nbits) * tile_h * tile_w * common_dim


def block_bound(header_nbits: int, tile_h: int, tile_w: int, common_dim: int) -> int:
    """The bound for an actual block solution (the header's own nbits)."""
    return nbits_to_difficulty(header_nbits) * tile_h * tile_w * common_dim


def meets_bound(hash_jackpot_le: bytes, bound: int) -> bool:
    """True iff U256_LE(hash_jackpot) <= bound (Pearl's nested-target check)."""
    if len(hash_jackpot_le) != 32:
        raise ValueError("hash_jackpot must be 32 bytes")
    return int.from_bytes(hash_jackpot_le, "little") <= bound


def parse_notify_target(target_hex: str) -> int:
    """Parse the 64-char big-endian hex ``target`` field of mining.notify."""
    t = target_hex.strip().lower().removeprefix("0x")
    if len(t) != 64:
        raise ValueError(f"target must be 64 hex chars, got {len(t)}")
    return int(t, 16)
