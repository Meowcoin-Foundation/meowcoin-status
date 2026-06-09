"""Gold-standard test: shares mined by meowminer engines must be accepted by
the actual Rust reference verifier (pearl_mining.verify_plain_proof).

Requires the pearl_mining wheel (and torch for the torch engine test); both
tests are skipped when the native deps are absent so the rest of the suite
stays runnable anywhere. Run on Python >= 3.12.

Validated constraints discovered against the verifier (2026-06-09):
  * k must be >= 16 * rank        ("k must be >= 16r")
  * tile h*w must be >= 32        ("Inner hash size must be >= 32")
  * MMAType on the network: Int7xInt7ToInt32
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

pm = pytest.importorskip("pearl_mining")

from meowminer.algo import MiningConfig, PeriodicPattern, compute_job_key  # noqa: E402
from meowminer.job import Job  # noqa: E402
from meowminer.proofs import build_plain_proof_b64  # noqa: E402
from meowminer.targets import nbits_to_difficulty  # noqa: E402

EASY_NBITS = 0x207FFFFF  # regtest-style: every attempt clears the bound


def make_fixture():
    rows = pm.PeriodicPattern.from_list(list(range(8)))  # 8x8 tile (>= 32 elems)
    cfg_pm = pm.MiningConfiguration(
        2048, 128, pm.MMAType.Int7xInt7ToInt32, rows, rows, bytes(pm.MiningConfiguration.RESERVED)
    )
    header_bytes = (
        bytes([1, 0, 0, 0]) + bytes(32) + bytes(32)
        + (0x66666666).to_bytes(4, "little") + EASY_NBITS.to_bytes(4, "little")
    )
    my_cfg = MiningConfig(
        common_dim=2048,
        rank=128,
        rows_pattern=PeriodicPattern(shape=tuple((int(s), int(l)) for s, l in rows.shape)),
        cols_pattern=PeriodicPattern(shape=tuple((int(s), int(l)) for s, l in rows.shape)),
        config_bytes=bytes(cfg_pm.to_bytes()),
    )
    job = Job(job_id="t", header=header_bytes, share_target=nbits_to_difficulty(EASY_NBITS), height=1)
    return header_bytes, my_cfg, job


def mine_one_and_verify(engine, my_cfg, job, header_bytes):
    bound = job.block_bound(my_cfg.tile_h, my_cfg.tile_w, my_cfg.dot_product_length)
    for hits, stats in engine.mine(job, bound, bound):
        assert stats.partitions == 256
        assert hits, "easy bound must hit on every partition"
        hit = hits[0]
        b64 = build_plain_proof_b64(hit, my_cfg, compute_job_key(header_bytes, my_cfg.config_bytes))
        header = pm.IncompleteBlockHeader.from_bytes(header_bytes)
        ok, msg = pm.verify_plain_proof(header, pm.PlainProof.from_base64(b64))
        assert ok, f"reference verifier rejected the share: {msg}"
        return


def test_cpu_engine_share_accepted_by_reference_verifier():
    pytest.importorskip("torch")  # vendored noise generator needs it
    from meowminer.engines.cpu import CpuEngine

    header_bytes, my_cfg, job = make_fixture()
    eng = CpuEngine(seed=1234)
    eng.configure(my_cfg, 128, 128)
    mine_one_and_verify(eng, my_cfg, job, header_bytes)


def test_torch_engine_share_accepted_by_reference_verifier():
    pytest.importorskip("torch")
    from meowminer.engines.torch_int8 import TorchInt8Engine

    header_bytes, my_cfg, job = make_fixture()
    eng = TorchInt8Engine(device="cpu", m=128, n=128)
    eng.configure(my_cfg)
    mine_one_and_verify(eng, my_cfg, job, header_bytes)
