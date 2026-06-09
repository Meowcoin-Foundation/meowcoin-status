"""meowminer CLI.

Example (herominers):
    meowminer --pool stratum+tcp://pearl.herominers.com:1225 \
              --wallet prl1p... --worker rig1 --engine torch --device cuda:0
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from urllib.parse import urlparse

from . import __version__
from .miner import Miner
from .netconfig import load_config
from .stratum import StratumClient


def parse_pool(pool: str) -> tuple[str, int]:
    if "//" not in pool:
        pool = "stratum+tcp://" + pool
    u = urlparse(pool)
    if not u.hostname or not u.port:
        raise SystemExit(f"--pool must be host:port or stratum+tcp://host:port (got {pool!r})")
    return u.hostname, u.port


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="meowminer", description="MeowMiner — Pearl (PRL) pool miner, no dev fee")
    p.add_argument("--pool", required=True, help="stratum endpoint, e.g. stratum+tcp://pearl.herominers.com:1225")
    p.add_argument("--wallet", required=True, help="PRL wallet address (prl1...)")
    p.add_argument("--worker", default="meowminer", help="worker name shown on the pool")
    p.add_argument("--engine", choices=["torch", "cpu"], default="torch")
    p.add_argument("--device", default="cuda", help="torch device for the GPU engine (cuda, cuda:1, ...)")
    p.add_argument("--matrix-m", type=int, default=4096, help="rows of A per attempt")
    p.add_argument("--matrix-n", type=int, default=4096, help="cols of B per attempt")
    p.add_argument("--dialect", choices=["object", "positional"], default="object",
                   help="stratum dialect: object = herominers/luckypool, positional = alphapool")
    p.add_argument("--config-bytes", default=None,
                   help="hex of the 52-byte network MiningConfiguration (default: pearl_mining default)")
    p.add_argument("--no-presubmit-verify", action="store_true",
                   help="skip local verify_plain_proof before submitting (not recommended)")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--version", action="version", version=f"meowminer {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    host, port = parse_pool(args.pool)
    config = load_config(args.config_bytes)

    if args.engine == "torch":
        from .engines.torch_int8 import TorchInt8Engine

        engine = TorchInt8Engine(device=args.device, m=args.matrix_m, n=args.matrix_n)
    else:
        from .engines.cpu import CpuEngine

        engine = CpuEngine()
    engine.configure(config, args.matrix_m, args.matrix_n)

    client = StratumClient(host, port, wallet=args.wallet, worker=args.worker,
                           agent=f"meowminer/{__version__}", dialect=args.dialect)
    miner = Miner(client, engine, config, presubmit_verify=not args.no_presubmit_verify)
    try:
        asyncio.run(miner.run())
    except KeyboardInterrupt:
        logging.getLogger("meowminer").info("interrupted; shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
