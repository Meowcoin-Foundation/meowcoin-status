# HANDOFF — Beat SRBMiner poolside on herominers (Pearl PRL, 4070 Ti Super)

**For:** the Claude session running on the desktop next to CatStack/Mfarm
(this work was done in a cloud sandbox that cannot reach the rigs — see
"Environment constraints" at the bottom).

**Mission:** poolside hashrate on herominers for the 4070 Ti Super rigs,
mining Pearl (PRL), higher than SRBMiner-Multi achieves on the same cards.

---

## 1. Where things stand

* `Meowcoin-Foundation/MeowMiner` existed as an **empty repo** with only the
  description "Pearl (PRL) sm_89-optimized GPU miner, no dev fee". This
  directory (`meowminer/`) is its initial implementation — **move it there**
  (`git subtree` or plain copy), rename `.github-workflows/` →
  `.github/workflows/`, and tag `v0.1.0` so CI publishes the rig bundle that
  `mfarm`'s miner-downloader expects.
* Everything that can be validated without a GPU **is validated**: 16 tests
  pass (target math, pattern partitioning, jackpot folding bit-exact vs a
  scalar port of zk-pow's `try_mine_one`, stratum client against an
  in-process pool speaking the P2Pearl object dialect) — including the gold
  standard, `tests/test_cross_engine.py`: **shares mined by both engines are
  accepted by the actual Rust reference verifier** (`verify_plain_proof`).
  Consensus-critical semantics (noise determinism, commitment seed chain,
  Merkle/chunk layout, jackpot folding, bincode wire format) are confirmed
  correct, not just believed correct.
* **End-to-end pool test passed**: the real `meowminer` CLI mined against
  P2Pearl's actual stratum server (the implementation validated against a
  real SRBMiner GPU on regtest) — connected in the herominers/luckypool
  object dialect, received jobs, and had every submitted 44 KB plain proof
  accepted after pool-side Rust verification (1-4 ms each), zero rejects.
* Sandbox CPU sanity bench at production dimensions (m=n=k=4096, reference
  2x64 tile, 131072 partitions/attempt): ~8 GH/s — the pipeline holds up at
  real sizes; the GPU run is purely a perf unknown, not a correctness one.
* `pearl_mining` (PyO3 bindings from pearl-research-labs/pearl) builds clean
  with maturin on Python 3.12; MeowMiner uses it for share packaging and
  pre-submit verification, so wire-format bugs are structurally impossible.

## 2. The competitive landscape (verified 2026-06-09)

| Miner | sm_89 hashrate | Fee | Notes |
|---|---|---|---|
| SRBMiner-Multi ≥3.3.1 (`--algorithm pearlhash`) | ~baseline | **3.00%** | the one to beat; poolside loses the fee on top of kernel quality |
| alpha-miner 1.7.7 | 4070: 65–70 TH/s, 4080S: 155–165 TH/s → 4070 TiS ≈ 115–130 TH/s | undocumented | binary-only; `--pool` accepts any stratum, marketed for AlphaPool |
| Pearl reference vLLM miner | n/a | 0 | **sm_90 only** (H100/H200) — why this niche exists |
| MeowMiner v0 (torch engine) | ~5–15 TH/s expected (CPU-hash-bound) | 0 | correctness vehicle; v1 kernel work is the win condition |

## 3. Pearlhash, the 60-second version (all file refs = pearl monorepo)

1. Job: pool sends a 76-byte incomplete header + 256-bit share `target`
   (`mining.notify {job_id, header, target, height}`; P2Pearl
   `stratum/server.py:87-95`). No nonce — the search space is the random
   int7 A (m×k) and B (k×n) matrices.
2. `job_key = blake3(header || mining_config)`; commitment seeds chain
   `hash_b → b_noise_seed → a_noise_seed` (`zk-pow/src/ffi/mine.rs:163-186`).
3. Deterministic low-rank noise (rank 128) is added to A/B
   (`miner/miner-base/.../noise_generation.py`, vendored bit-exact in
   `src/meowminer/vendor/`).
4. GEMM with per-rank-block cumulative tiles; each (rows_pattern ×
   cols_pattern) partition XOR-folds its cumulative tile into a 16×u32
   jackpot with rotl-13 mixing (`mine.rs:78-117`,
   `pearl_program.rs: JACKPOT_SIZE=16, LROT_PER_TILE=13`).
5. `hash_jackpot = blake3(jackpot_le_bytes, key=a_noise_seed)`; share iff
   `U256_LE(hash_jackpot) <= nbits_to_difficulty(target_to_bits(target)) * h*w*k`.
   **The h\*w\*k (~2^19–2^20) verifier adjustment is the classic landmine** —
   grade at the raw notify target and you throw away ~all your shares
   (`zk-pow/src/api/sanity_checks.rs:86-102`, P2Pearl `pow/verify.py`).
6. Submit `mining.submit {job_id, plain_proof: base64(bincode(PlainProof)), hs}`
   — Merkle-authenticated strips of the *unnoised* A/B rows (137–370 KB).
   Pools verify CPU-side with `verify_plain_proof_with_nbits`; no ZK proof
   for shares (only for found blocks, gateway-side).

The "hashrate" everyone quotes = MACs/s against graded jackpots
(= m·n·k per attempt when patterns exactly cover the matrices), which is why
4070 Ti Super (~353 dense int8 TOPS) tops out near ~150 TH/s.

## 4. The plan to win, in order

### Step 0 — confirm pool specifics (15 min, desktop)
RESOLVED: the MiningConfiguration does NOT need network pinning — the verifier
reconstructs it from the submitted proof (patterns via `list_to_pattern(row
indices)`, k/rank from proof fields; zk-pow/src/ffi/plain_proof.rs::
parse_plain_proof), so the miner picks its own dimensions and share economics
are dimension-neutral. `netconfig.py` defaults to the reference miner's 2x64
hash tile with k=4096. Verifier-enforced sanity limits (confirmed empirically
against pearl_mining): `k >= 16*rank`, tile `h*w >= 32`, MMAType
`Int7xInt7ToInt32`, signal in [-64, 64].
Still to confirm on the desktop: the herominers PRL stratum port (their site
is blocked from this sandbox) and the payout wallet (see bottom).

### Step 1 — first light on a 4070 Ti Super (hours)
```
pip install -e meowminer[gpu]; pip install pearl_mining-*.whl
meowminer --pool stratum+tcp://pearl.herominers.com:<port> --wallet prl1p... \
          --worker bench1 --engine torch --log-level DEBUG
```
The built-in pre-submit `verify_plain_proof_with_nbits` check means: if shares
pass locally and the pool still rejects, the dialect needs a tweak (capture
SRBMiner's traffic with `socat -v` as ground truth); if they fail locally,
the engine has a bug — `tests/test_cross_engine.py` is the harness to extend
(compare torch engine vs `engines/cpu.py` on the same RNG seed).

### Step 2 — A/B against SRBMiner via Mfarm (1 day)
`mfarm-integration/FLIGHT-SHEET.md` has the exact commands: half the 4070 TiS
group on MeowMiner, half on SRBMiner, same wallet, 24h, compare per-worker
poolside averages on herominers. This is the metric the user cares about.

### Step 3 — v1 kernel work: move the hashing onto the GPU (the actual win)
v0's ceiling is the CPU blake3 of the A/B commitment (the GEMM:hash byte
ratio is m·n/(m+n) — you cannot out-batch it). The reference already has GPU
kernels for everything needed:

* `miner/pearl-gemm/csrc/blake3/blake3.cu` — portable CUDA blake3 (~209 lines).
* `miner/pearl-gemm/csrc/tensor_hash/` — chunked Merkle/commitment pipeline
  (~2k lines; entangled with cute/cutlass *types* but not with WGMMA — the
  compute is hash arithmetic, port to plain CUDA or build with
  `-gencode arch=compute_89,code=sm_89`).
* Keep the GEMM itself on cuBLASLt int8 (`torch._int_mm`) — on Ada it reaches
  80%+ of peak TOPS for 4096-class shapes, which already clears alpha-miner's
  published numbers; the full CUTLASS sm_89 NoisyGEMM port
  (`kernel_traits.hpp` TMA/WGMMA → cp.async/mma.sync) is optional polish.
* Wire them behind the `hasher` seam in `engines/torch_int8.py` (the seam is
  already there), JIT via `torch.utils.cpp_extension.load` on rigs with nvcc
  or prebuild in CI.

Target after step 3: ≥130 TH/s poolside per 4070 Ti Super, vs SRBMiner's
baseline minus its 3% fee.

### Step 4 — publish + farm rollout
Tag MeowMiner `v0.2.0`; CI bundles wheels; `mfarm deploy` + flight-sheet flip
moves the whole 4070 TiS group. Keep one SRBMiner rig as a control.

## 5. Map of this directory

```
meowminer/
  src/meowminer/         the package (stratum, targets, algo, engines, proofs, cli)
  src/meowminer/vendor/  bit-exact vendored noise generator (ISC, pearl)
  tests/                 all green on CPU-only machines
  mfarm-integration/     CatStack patch + flight sheets + A/B protocol
  packaging/install.sh   rig-side bundle installer (used by miner-downloader)
  .github-workflows/     CI for the MeowMiner repo (rename to .github/workflows)
  HANDOFF.md             this file
```

## 6. Environment constraints that shaped this session

Cloud sandbox: GitHub+PyPI only (every other domain, SSH, and the user's LAN
blocked — verified with curl, headless Chromium, and raw TCP). CatStack's
dashboard lives on the desktop at localhost:8888 and the rigs are LAN-side,
so deployment and GPU benchmarking are physically impossible from here; hence
this handoff. Wallet (herominers/pearlhash notes from self-sent emails):
the AlphaPool one is `prl1pja266dfa7kcg0xdagaacy0y7x60h7qrw3tcau4enx4gwnmmyxxvs7ep7ad`,
the Pearlhash-pool one is `prl1p6zlg6r00gv3tqhauc3esz3mx8fdzuypn6w6plywx98x4ntfaraqs9pmvr6`
— confirm which one herominers payouts should go to before creating flight
sheets.
