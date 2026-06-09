"""Orchestrator: stratum jobs in, engine batches out, shares submitted."""

from __future__ import annotations

import asyncio
import logging
import threading

from .algo import MiningConfig, compute_job_key
from .engines.base import Engine, Hit
from .job import Job
from .proofs import build_plain_proof_b64, verify_share_locally
from .stats import MinerStats
from .stratum import StratumClient
from .targets import target_to_bits

log = logging.getLogger("meowminer")

STATS_INTERVAL_S = 30.0


class Miner:
    def __init__(self, client: StratumClient, engine: Engine, config: MiningConfig, presubmit_verify: bool = True) -> None:
        self.client = client
        self.engine = engine
        self.config = config
        self.presubmit_verify = presubmit_verify
        self.stats = MinerStats()
        self._job: Job | None = None
        self._job_seq = 0  # bumped on every new job; mining thread watches it
        self._job_lock = threading.Lock()
        self._stop = False

    # -- lifecycle -------------------------------------------------------------

    async def run(self) -> None:
        self.client.on_job = self._on_job
        await self.client.connect()
        loop = asyncio.get_running_loop()
        mining = loop.run_in_executor(None, self._mining_thread, loop)
        stats_task = asyncio.create_task(self._stats_loop())
        try:
            await mining
        finally:
            self._stop = True
            stats_task.cancel()
            await self.client.close()

    async def _on_job(self, job: Job) -> None:
        with self._job_lock:
            self._job = job
            self._job_seq += 1

    async def _stats_loop(self) -> None:
        while not self._stop:
            await asyncio.sleep(STATS_INTERVAL_S)
            log.info("%s", self.stats.summary())

    # -- mining thread ----------------------------------------------------------

    def _mining_thread(self, loop: asyncio.AbstractEventLoop) -> None:
        import time

        while not self._stop:
            with self._job_lock:
                job, seq = self._job, self._job_seq
            if job is None:
                time.sleep(0.2)
                continue

            share_bound = job.share_bound(self.config.tile_h, self.config.tile_w, self.config.dot_product_length)
            block_bound = job.block_bound(self.config.tile_h, self.config.tile_w, self.config.dot_product_length)
            log.info(
                "mining job %s: share_bound=%x... (%d-bit)",
                job.job_id, share_bound >> 192, share_bound.bit_length(),
            )

            for hits, stats in self.engine.mine(job, share_bound, block_bound):
                self.stats.record(stats.macs)
                for hit in hits:
                    self._submit(hit, loop)
                with self._job_lock:
                    if self._job_seq != seq or self._stop:
                        break  # job changed: tear down iterator, restart loop

    def _submit(self, hit: Hit, loop: asyncio.AbstractEventLoop) -> None:
        job_key = compute_job_key(hit.job.header, self.config.config_bytes)
        try:
            proof_b64 = build_plain_proof_b64(hit, self.config, job_key)
        except Exception:
            log.exception("failed to package share; dropping")
            return

        if self.presubmit_verify:
            share_nbits = target_to_bits(hit.job.share_target)
            ok, msg = verify_share_locally(hit.job.header, proof_b64, share_nbits)
            if not ok:
                log.error("local share verification FAILED (%s) — not submitting. "
                          "This means an engine correctness bug; please report.", msg)
                return

        if hit.is_block:
            log.warning("*** BLOCK CANDIDATE *** job=%s hash=%s", hit.job.job_id, hit.hash_jackpot.hex()[:24])
            self.stats.blocks += 1

        async def do_submit():
            res = await self.client.submit(hit.job.job_id, proof_b64, hashrate=self.stats.hashrate())
            if res.accepted:
                self.stats.accepted += 1
            elif res.error_code == 21:
                self.stats.stale += 1
            else:
                self.stats.rejected += 1

        asyncio.run_coroutine_threadsafe(do_submit(), loop)
