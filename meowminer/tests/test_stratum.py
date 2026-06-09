"""Stratum client against an in-process pool speaking the P2Pearl object dialect."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meowminer.job import Job
from meowminer.stratum import StratumClient

HEADER_HEX = ("01000000" + "11" * 32 + "22" * 32 + "66666666" + "ffff7f20")
TARGET_HEX = "0000000000000000000000000000000000000000000000000000000000010000"


class FakePool:
    def __init__(self):
        self.received = []
        self.server = None
        self.port = None
        self.accept_shares = True

    async def start(self):
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def _handle(self, reader, writer):
        async def send(obj):
            writer.write((json.dumps(obj, separators=(",", ":")) + "\n").encode())
            await writer.drain()

        while True:
            line = await reader.readline()
            if not line:
                return
            msg = json.loads(line)
            self.received.append(msg)
            method = msg.get("method")
            if method == "mining.authorize":
                await send({"id": msg["id"], "result": True})
                await send({
                    "id": None,
                    "method": "mining.notify",
                    "params": {"job_id": "2a-1", "header": HEADER_HEX, "target": TARGET_HEX, "height": 42},
                })
            elif method == "mining.submit":
                if self.accept_shares:
                    await send({"id": msg["id"], "result": True})
                else:
                    await send({"id": msg["id"], "result": None, "error": [23, "low difficulty share", None]})
            else:
                await send({"id": msg.get("id"), "result": True})


def test_object_dialect_end_to_end():
    async def scenario():
        pool = FakePool()
        await pool.start()

        jobs: list[Job] = []
        got_job = asyncio.Event()

        client = StratumClient("127.0.0.1", pool.port, wallet="prl1ptest", worker="rigtest")

        async def on_job(job: Job):
            jobs.append(job)
            got_job.set()

        client.on_job = on_job
        await client.connect()
        await asyncio.wait_for(got_job.wait(), 5)

        job = jobs[0]
        assert job.job_id == "2a-1"
        assert job.height == 42
        assert job.share_target == 0x10000
        assert len(job.header) == 76
        assert job.nbits == 0x207FFFFF  # LE bytes ffff7f20
        assert job.timestamp == 0x66666666

        ok = await client.submit("2a-1", "cHJvb2Y=", hashrate=123.0)
        assert ok.accepted

        pool.accept_shares = False
        bad = await client.submit("2a-1", "cHJvb2Y=")
        assert not bad.accepted
        assert bad.error_code == 23

        auth = pool.received[0]
        assert auth["method"] == "mining.authorize"
        assert auth["params"] == {"wallet": "prl1ptest", "worker": "rigtest", "agent": "meowminer/0.1.0"}
        submit = next(r for r in pool.received if r.get("method") == "mining.submit")
        assert submit["params"]["job_id"] == "2a-1"
        assert submit["params"]["plain_proof"] == "cHJvb2Y="
        assert submit["params"]["hs"] == 123.0

        await client.close()

    asyncio.run(scenario())
