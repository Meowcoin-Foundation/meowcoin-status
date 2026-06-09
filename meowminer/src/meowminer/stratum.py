"""Asyncio stratum client for Pearlhash pools.

Speaks the "object" dialect used by herominers / luckypool (the one SRBMiner
speaks), with the alphapool "positional" dialect as a fallback:

  object:
    -> mining.authorize {"wallet": "prl1...", "worker": "rig", "agent": "meowminer/x.y"}
    <- mining.notify    {"job_id", "header", "target", "height"[, "clean"]}
    -> mining.submit    {"job_id", "plain_proof": "<b64>", "hs": <float>}

  positional (alphapool):
    -> mining.subscribe ["meowminer/x.y"]
    -> mining.authorize ["<wallet>.<worker>", "x"]
    <- mining.notify    [job_id, prev_hash, header, 0, ntime, nbits, clean]
    -> mining.submit    [worker, job_id, "<b64>"]

Framing: newline-delimited JSON; responses are bare {"id", "result"} with no
"jsonrpc" member; notifications carry "id": null.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from .job import Job

log = logging.getLogger("meowminer.stratum")

STALE_SHARE_CODE = 21
LOW_DIFF_CODE = 23


@dataclass
class SubmitResult:
    accepted: bool
    error_code: int | None = None
    message: str = ""


class StratumClient:
    def __init__(
        self,
        host: str,
        port: int,
        wallet: str,
        worker: str = "meowminer",
        agent: str = "meowminer/0.1.0",
        dialect: str = "object",
        timeout: float = 30.0,
    ) -> None:
        self.host, self.port = host, port
        self.wallet, self.worker, self.agent = wallet, worker, agent
        self.dialect = dialect
        self.timeout = timeout
        self.on_job: Callable[[Job], Awaitable[None]] | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 1
        self._read_task: asyncio.Task | None = None
        self.connected = asyncio.Event()

    # -- connection lifecycle -------------------------------------------------

    async def connect(self) -> None:
        backoff = 1.0
        while True:
            try:
                self._reader, self._writer = await asyncio.wait_for(
                    asyncio.open_connection(self.host, self.port), self.timeout
                )
                break
            except (OSError, asyncio.TimeoutError) as exc:
                log.warning("connect to %s:%s failed (%s); retrying in %.0fs", self.host, self.port, exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
        self._read_task = asyncio.create_task(self._read_loop())
        await self._handshake()
        self.connected.set()
        log.info("connected to %s:%s as %s.%s (%s dialect)", self.host, self.port, self.wallet[:14] + "...", self.worker, self.dialect)

    async def close(self) -> None:
        self.connected.clear()
        if self._read_task:
            self._read_task.cancel()
        if self._writer:
            self._writer.close()

    async def _handshake(self) -> None:
        if self.dialect == "positional":
            await self._call("mining.subscribe", [self.agent])
            result = await self._call("mining.authorize", [f"{self.wallet}.{self.worker}", "x"])
        else:
            result = await self._call(
                "mining.authorize",
                {"wallet": self.wallet, "worker": self.worker, "agent": self.agent},
            )
        if result is not True and result is not None:
            log.warning("authorize returned %r (continuing; some pools reply oddly)", result)

    # -- rpc ------------------------------------------------------------------

    async def _send(self, payload: dict) -> None:
        assert self._writer is not None
        self._writer.write((json.dumps(payload, separators=(",", ":")) + "\n").encode())
        await self._writer.drain()

    async def _call(self, method: str, params) -> object:
        req_id = self._next_id
        self._next_id += 1
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        await self._send({"id": req_id, "method": method, "params": params})
        try:
            return await asyncio.wait_for(fut, self.timeout)
        finally:
            self._pending.pop(req_id, None)

    async def _read_loop(self) -> None:
        assert self._reader is not None
        try:
            while True:
                line = await self._reader.readline()
                if not line:
                    raise ConnectionError("pool closed connection")
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("unparseable line from pool: %r", line[:200])
                    continue
                await self._dispatch(msg)
        except (ConnectionError, asyncio.IncompleteReadError) as exc:
            log.warning("pool connection lost: %s", exc)
            self.connected.clear()
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("disconnected"))
        except asyncio.CancelledError:
            pass

    async def _dispatch(self, msg: dict) -> None:
        if msg.get("method") == "mining.notify":
            try:
                job = self._parse_notify(msg.get("params"))
            except (KeyError, ValueError) as exc:
                log.error("bad mining.notify: %s (params=%r)", exc, msg.get("params"))
                return
            log.info("new job %s height=%s target=%064x clean=%s", job.job_id, job.height, job.share_target, job.clean)
            if self.on_job:
                await self.on_job(job)
            return
        if msg.get("method") == "mining.set_target":
            # Some pools retarget without a fresh notify; surface as a job-less event.
            log.info("pool set_target: %r", msg.get("params"))
            return
        req_id = msg.get("id")
        fut = self._pending.get(req_id)
        if fut is not None and not fut.done():
            if msg.get("error"):
                fut.set_exception(StratumError(msg["error"]))
            else:
                fut.set_result(msg.get("result"))
        elif msg.get("method"):
            log.debug("ignoring server method %s", msg["method"])

    def _parse_notify(self, params) -> Job:
        if isinstance(params, dict):
            return Job.from_notify(params)
        if isinstance(params, list) and len(params) >= 7:
            # alphapool positional: [job_id, prev_hash, header, 0, ntime, nbits, clean]
            import struct as _struct

            header = bytes.fromhex(params[2])
            nbits = int(params[5], 16) if isinstance(params[5], str) else int(params[5])
            from .targets import nbits_to_difficulty

            return Job(
                job_id=str(params[0]),
                header=header,
                share_target=nbits_to_difficulty(nbits),
                height=0,
                clean=bool(params[6]),
            )
        raise ValueError("unrecognized mining.notify shape")

    # -- shares ---------------------------------------------------------------

    async def submit(self, job_id: str, plain_proof_b64: str, hashrate: float | None = None) -> SubmitResult:
        if self.dialect == "positional":
            params = [self.worker, job_id, plain_proof_b64]
        else:
            params = {"job_id": job_id, "plain_proof": plain_proof_b64}
            if hashrate is not None:
                params["hs"] = round(hashrate, 2)
        t0 = time.monotonic()
        try:
            result = await self._call("mining.submit", params)
        except StratumError as exc:
            code = exc.code
            label = {STALE_SHARE_CODE: "stale", LOW_DIFF_CODE: "low difficulty"}.get(code, "rejected")
            log.warning("share REJECTED (%s): %s", label, exc)
            return SubmitResult(False, code, str(exc))
        log.info("share accepted in %.0f ms", (time.monotonic() - t0) * 1e3)
        return SubmitResult(bool(result))


class StratumError(Exception):
    def __init__(self, error) -> None:
        code, message = None, str(error)
        if isinstance(error, list) and len(error) >= 2:
            code, message = error[0], str(error[1])
        super().__init__(message)
        self.code = code
