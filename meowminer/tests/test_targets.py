"""Target math vectors, cross-checked against P2Pearl/zk-pow semantics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meowminer.targets import (
    meets_bound,
    nbits_to_difficulty,
    parse_notify_target,
    share_bound,
    target_to_bits,
)


def test_nbits_roundtrip_genesis_style():
    # 0x207fffff: classic max-target regtest nbits
    t = nbits_to_difficulty(0x207FFFFF)
    assert t == 0x7FFFFF << (8 * (0x20 - 3))
    assert target_to_bits(t) == 0x207FFFFF


def test_nbits_invalid_encodings():
    assert nbits_to_difficulty(0) == 0
    assert nbits_to_difficulty(0x00FFFFFF) == 0  # zero exponent
    assert nbits_to_difficulty(0x20800000) == 0  # sign bit set
    assert nbits_to_difficulty(0x01000000) == 0  # zero mantissa


def test_nbits_small_exponents():
    # exponent <= 3 shifts the 3-byte mantissa right by 8*(3-exponent)
    assert nbits_to_difficulty(0x01120000) == 0x120000 >> 16  # 0x12
    assert nbits_to_difficulty(0x02123400) == 0x123400 >> 8  # 0x1234
    assert nbits_to_difficulty(0x03123456) == 0x123456
    assert nbits_to_difficulty(0x04123456) == 0x123456 << 8


def test_target_to_bits_sign_bit_normalization():
    # A target whose top mantissa byte has the high bit set must be renormalized
    t = 0x80FFFF << 8
    nbits = target_to_bits(t)
    assert not (nbits & 0x00800000)
    # round-trips within compact precision
    assert target_to_bits(nbits_to_difficulty(nbits)) == nbits


def test_share_bound_applies_hwk_adjustment():
    target = 1 << 200
    h, w, k = 16, 16, 4096
    b = share_bound(target, h, w, k)
    base = nbits_to_difficulty(target_to_bits(target))
    assert b == base * h * w * k
    assert b > target  # the adjustment makes the bound ~2**20 easier


def test_meets_bound_little_endian():
    # hash bytes are compared as a LITTLE-endian integer
    h = bytes([1] + [0] * 31)  # LE value 1
    assert meets_bound(h, 1)
    assert not meets_bound(h, 0)
    h2 = bytes([0] * 31 + [1])  # LE value 2**248
    assert not meets_bound(h2, 1 << 247)
    assert meets_bound(h2, 1 << 248)


def test_parse_notify_target():
    assert parse_notify_target("00" * 31 + "ff") == 0xFF
    assert parse_notify_target(("0" * 63) + "1") == 1
    try:
        parse_notify_target("ff")
    except ValueError:
        pass
    else:
        raise AssertionError("short target must raise")
