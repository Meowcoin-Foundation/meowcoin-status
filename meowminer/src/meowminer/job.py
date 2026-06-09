"""Mining job state derived from ``mining.notify``."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .targets import block_bound, parse_notify_target, share_bound

HEADER_LEN = 76  # version(4) | prev_block(32) | merkle_root(32) | timestamp(4) | nbits(4)


@dataclass(frozen=True)
class Job:
    """One unit of pool work (object dialect: herominers / luckypool).

    ``header`` is the 76-byte *incomplete* header exactly as the pool sent it;
    it feeds ``job_key = blake3(header || mining_config)`` and must never be
    mutated (the share target is communicated out-of-band, NOT via nbits).
    """

    job_id: str
    header: bytes
    share_target: int  # 256-bit integer from the notify `target` field
    height: int
    clean: bool = False

    def __post_init__(self) -> None:
        if len(self.header) != HEADER_LEN:
            raise ValueError(f"incomplete header must be {HEADER_LEN} bytes, got {len(self.header)}")

    @classmethod
    def from_notify(cls, params: dict) -> "Job":
        return cls(
            job_id=str(params["job_id"]),
            header=bytes.fromhex(params["header"]),
            share_target=parse_notify_target(params["target"]),
            height=int(params.get("height", 0)),
            clean=bool(params.get("clean", False)),
        )

    @property
    def nbits(self) -> int:
        return struct.unpack_from("<I", self.header, 72)[0]

    @property
    def timestamp(self) -> int:
        return struct.unpack_from("<I", self.header, 68)[0]

    def share_bound(self, tile_h: int, tile_w: int, common_dim: int) -> int:
        return share_bound(self.share_target, tile_h, tile_w, common_dim)

    def block_bound(self, tile_h: int, tile_w: int, common_dim: int) -> int:
        return block_bound(self.nbits, tile_h, tile_w, common_dim)
