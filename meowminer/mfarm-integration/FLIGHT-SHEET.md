# Deploying MeowMiner on the farm through Mfarm (CatStack)

Apply `catstack-meowminer.patch` to the CatStack checkout first
(`git apply mfarm-integration/catstack-meowminer.patch`), reinstall
(`pip install -e .`), then:

## 1. Flight sheets — herominers PRL

```bash
# MeowMiner on the 4070 Ti Super group
mfarm flight create prl-herominers-meowminer \
  --coin PRL --algo pearlhash --miner meowminer \
  --pool stratum+tcp://pearl.herominers.com:1225 \
  --wallet prl1p<YOUR_WALLET> \
  --worker %HOSTNAME% \
  --extra-args "--engine torch --device cuda --matrix-m 4096 --matrix-n 4096"

# SRBMiner baseline on the SAME pool/wallet for the A/B comparison
mfarm flight create prl-herominers-srb \
  --coin PRL --algo pearlhash --miner srbminer \
  --pool stratum+tcp://pearl.herominers.com:1225 \
  --wallet prl1p<YOUR_WALLET> \
  --worker %HOSTNAME%-srb \
  --extra-args "--disable-cpu"
```

Check the actual herominers PRL stratum port on https://pearl.herominers.com
before creating the sheets (1225 is a placeholder — this sandbox could not
reach the site to confirm). Use the high-difficulty port for >4 GPU rigs if
they offer one.

## 2. Overclocks for 4070 Ti Super on pearlhash

pearlhash is tensor-core int8 GEMM: core-clock and cache bound, only mildly
memory bound (operands are reused k times). Suggested starting profile:

```bash
mfarm oc create prl-4070tis --power-limit 270 --core-offset 150 --mem-offset 0 --fan 70
mfarm oc apply prl-4070tis group:4070tis
```

Then sweep power limit 240→285 W while watching poolside hashrate; pearlhash
typically keeps >95% hashrate down to ~80% TDP. Memory offset buys ~nothing;
leave it stock for stability.

## 3. The A/B test that settles "beat SRBMiner"

Split identical rigs into two halves for 24h (PPLNS smoothing window matters,
don't compare 10-minute spot values):

```bash
mfarm flight apply prl-herominers-meowminer group:4070tis-a
mfarm flight apply prl-herominers-srb       group:4070tis-b
```

Compare on the herominers dashboard per worker name (`rigN` vs `rigN-srb`):
*poolside average hashrate over 24h* and *accepted share count*. Remember
SRBMiner's pearlhash carries a 3.00% dev fee that shows up poolside as ~3%
missing shares; MeowMiner has none.

## 4. Watchdog

`mfarm-agent` restarts the miner on crash/zero-hashrate via the standard
wrapper; meowminer logs `... TH/s | A:n R:n S:n` lines that the agent's
`meowminer_log` parser picks up.
