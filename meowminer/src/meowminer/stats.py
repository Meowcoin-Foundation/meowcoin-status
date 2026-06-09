"""Hashrate accounting.

Reports the same unit the pools and hashrate.no use for pearlhash: hashes =
multiply-accumulates executed against graded jackpots (h*w*k per partition),
i.e. the number the `hs` submit field and poolside estimates are based on.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class MinerStats:
    window_s: float = 600.0
    started: float = field(default_factory=time.monotonic)
    accepted: int = 0
    rejected: int = 0
    stale: int = 0
    blocks: int = 0
    _samples: deque = field(default_factory=deque)  # (t, macs)
    _total_macs: int = 0

    def record(self, macs: int) -> None:
        now = time.monotonic()
        self._samples.append((now, macs))
        self._total_macs += macs
        cutoff = now - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def hashrate(self) -> float:
        """Windowed hashes/second."""
        if not self._samples:
            return 0.0
        t0 = self._samples[0][0]
        elapsed = max(time.monotonic() - t0, 1e-9)
        return sum(m for _, m in self._samples) / elapsed

    def summary(self) -> str:
        hr = self.hashrate()
        unit, scale = "H/s", 1.0
        for u, s in (("TH/s", 1e12), ("GH/s", 1e9), ("MH/s", 1e6), ("kH/s", 1e3)):
            if hr >= s:
                unit, scale = u, s
                break
        up = time.monotonic() - self.started
        return (
            f"{hr / scale:.2f} {unit} | A:{self.accepted} R:{self.rejected} S:{self.stale}"
            f"{' | BLOCKS:' + str(self.blocks) if self.blocks else ''} | up {up / 60:.0f}m"
        )
