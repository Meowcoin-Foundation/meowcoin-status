# MeowMiner

**Pearl (PRL) pool miner for NVIDIA GPUs. No dev fee.**

Pearl's proof-of-useful-work is verifiable int8 matrix multiplication
("pearlhash"): every mining attempt draws random A (m×k) / B (k×n) matrices,
adds deterministic header-bound noise, runs the GEMM on tensor cores while
XOR-folding cumulative tiles into a 64-byte *jackpot* transcript, and
blake3-hashes that transcript against the pool's share target. MeowMiner is a
clean-room pool client over the Pearl reference primitives
([pearl-research-labs/pearl](https://github.com/pearl-research-labs/pearl),
ISC), aimed at consumer Ada cards (sm_89: 4070/4070 Ti Super/4080) that the
reference vLLM miner (sm_90 only) does not support — and at the 3.00% dev fee
that SRBMiner charges for the same algorithm.

## Status

| Component | State |
|---|---|
| Stratum (herominers/luckypool "object" dialect + alphapool positional) | implemented, unit-tested |
| Target/bound math incl. the `h*w*k` verifier adjustment | implemented, unit-tested |
| Jackpot pipeline (noise, commitment chain, XOR/rotl folding) | implemented, bit-exact vs a scalar port of the Rust reference, unit-tested |
| Share packaging + pre-submit local verification | implemented via `pearl_mining` (PyO3) |
| CPU reference engine | working; **shares accepted by the Rust reference verifier** |
| GPU engine v0 (`torch._int_mm` tensor-core GEMM, GPU folding) | **shares accepted by the Rust reference verifier** (CPU device); needs its first perf run on a real GPU |
| GPU engine v1 (CUDA blake3/tensor-hash on-device, sm_89 build) | planned — see `HANDOFF.md`; this is what overtakes SRBMiner |

## Install (rig)

```bash
pip install meowminer[gpu]
# plus the pearl_mining wheel, built once from the Pearl monorepo:
#   cd pearl/py-pearl-mining && maturin build --release
pip install pearl_mining-*.whl
```

## Run

```bash
meowminer --pool stratum+tcp://pearl.herominers.com:<PORT> \
          --wallet prl1p<...> --worker rig1 \
          --engine torch --device cuda
```

Every share is verified locally with the reference verifier
(`verify_plain_proof_with_nbits`) before submission; a correctness bug shows
up as a loud local error, never as silent pool rejects.

## Farm deployment

See `mfarm-integration/` for the CatStack (Mfarm) registry patch, the
herominers flight sheets, and the SRBMiner A/B test procedure.

## License

ISC. Vendored reference code from pearl-research-labs/pearl is ISC; see
`src/meowminer/vendor/PEARL-LICENSE`.
