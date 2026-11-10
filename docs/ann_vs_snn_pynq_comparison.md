# ANN vs SNN on the PYNQ-Z2 RISC-V core: first hardware comparison

*Group 4: ANN vs SNN on MNIST with RISC-V. Measured 2026-10-01.*

This is the first on-hardware comparison of the two networks from
`code/ann_vs_snn_mnist/train_mnist.py`. Both run on the same PicoRV32
soft core, the same unchanged bitstream, with the same quantization scheme
and the same level of software optimisation. Earlier documents cover the
setup and the SNN port:
[snn_mnist_pynq_deployment.md](snn_mnist_pynq_deployment.md) and
[snn_mnist_pynq_full_testset.md](snn_mnist_pynq_full_testset.md).

---

## 1. Headline

| (mean ± std over seeds 0, 1; all 10 000 test images) | ANN | SNN |
|---|---|---|
| board test accuracy | 96.84 ± 0.16 % | **97.65 ± 0.11 %** |
| board test loss | 0.1090 ± 0.0031 | **0.0772 ± 0.0029** |
| board latency, optimised firmware | **13.62 ms** | 57.13 ± 0.10 ms |
| board latency, baseline firmware (1000 images) | **135.8 ms** | 402.7 ms |
| ops/sample as defined in `benchmark.py` | 101 632 MACs | 336 608 SOPs |
| ops/sample actually executed (optimised) | **20 167 MACs** | 106 793 adds |
| bit-exact vs PC int8 reference | 20 000 / 20 000 | 20 000 / 20 000 |

**On this platform the ANN is 4.2× faster than the SNN, on every single
test image. The SNN is 0.8 pp more accurate.** The ANN gains more from
sparsity than the SNN gains from being event-driven: only 151 of 784 pixels
are nonzero, whereas the SNN has to simulate 25 timesteps of neuron state.

**External reference (added 2026-10-07):** the same int8 ANN on the
unmodified muRISCV-NN library (RISC-V CMSIS-NN, scalar C) takes 76.6 ms per
image. That is 1.8× faster than our baseline ANN, but slower than our fast
SNN on 9 998 of 10 000 images. The ANN's lead comes from exploiting
sparsity, which a standard dense library does not do. See
[muriscv_reference.md](muriscv_reference.md).

---

## 2. What was compared, and how fair it is

| held equal | how |
|---|---|
| network | 784-128-10, 101 770 parameters, same training budget (`train_mnist.py`, 5 epochs, seeds 0 and 1) |
| test data | the same 10 000 MNIST test images, in the same order (per image paired results) |
| hardware | PicoRV32 RV32IM @ 100 MHz, unchanged `spike_top.bit`, 256 KB BRAM, 8-wait-state bus |
| software | **one** firmware image runs both models (type from the `model.bin` magic); same compiler and flags |
| weights | int8, symmetric, one scale per layer, int32 biases, same 103 024 B model layout |
| correctness | each board output (SNN spike counts / ANN logits) equals the PC int8 model **bit for bit** |
| optimisation effort | two builds each: a straightforward baseline and the same set of optimisations (§3) |

**What necessarily differs:**

- **Inputs.** The SNN gets the rate-encoded spike trains of `benchmark.py`:
  T = 25 bitmaps, 2500 B per image. The ANN gets the raw uint8 pixels,
  784 B per image. This is exact, because `ToTensor` = pixel / 255.
- **Hidden activations.** In the SNN they are binary spikes, so no
  requantization is needed. In the ANN they are uint8, via
  `min(255, (relu(acc)·m_q + 2²³) >> 24)`, with the scale calibrated on
  10 000 training images (max hidden activation 13.0).
- **Readout.** SNN: argmax of the output spike counts. ANN: argmax of the
  int32 logits.

Quantization costs nothing measurable for either model. Full test set,
float → int8: ANN 96.95 → 96.95 % / 96.69 → 96.73 %, SNN 97.69 → 97.72 % /
97.59 → 97.57 %. So the accuracy difference between the models comes from
training, not from the port.

---

## 3. The two firmware levels

| | ANN | SNN |
|---|---|---|
| **baseline** | dense textbook layer: for every input (zeros included) × every output, `lb + mul + add`, accumulator in memory | event-driven by nature: per input spike, add the weight row byte by byte (`lb, lw, add, sw`) |
| **fast: sparsity** | skip zero pixels / zero ReLU outputs (151 of 784 pixels, 82 of 128 hidden on average) | **delta currents**: only inputs that changed since the last step (802 of 2 598 spikes per image), exact |
| **fast: SWAR** | 1 load = 4 int8 weights (offset binary); even/odd bytes → two 16-bit lanes; **one `mul` scales two weights** (255·255 < 2¹⁶) | 1 load = 4 int8 weights; even/odd bytes added into 16-bit lanes (flushed every ≤ 256 rows) |
| **fast: blocking** | 8 outputs per pass in registers (16 spilled to the stack, checked in the disassembly) | 16 outputs per pass in 8 registers (2 lanes each) |
| instructions per synaptic op (inner loop) | ~3.5 per MAC | ~1.5 per add |

Both kernels are in `code/ann_vs_snn_mnist/pynq/firmware/` (`ann.c`,
`main.c`). They are selected with `--fw baseline|fast`.

---

## 4. Results

### 4.1 Benchmark table (optimised firmware, full test set)

Output of `python pynq/benchmark_pynq.py` (`runs/summary_pynq_fast.md`):

| metric (mean ± std over seeds) | ANN | SNN |
|---|---|---|
| board test accuracy [%] | 96.84 ± 0.16 | 97.65 ± 0.11 |
| board test loss | 0.1090 ± 0.0031 | 0.0772 ± 0.0029 |
| PC float accuracy, same images [%] | 96.82 ± 0.18 | 97.64 ± 0.07 |
| board == PC int8 reference [%] | 100.00 | 100.00 |
| parameters | 101 770 | 101 770 |
| ops/sample, `benchmark.py` (MAC or SOP) | 101 632 | 336 608 ± 402 |
| ops/sample executed | 20 167 ± 1 | 106 793 ± 376 |
| nonzero inputs / input spikes per image | 151.1 | 2 597.5 |
| nonzero hidden / hidden spikes per image | 82.4 ± 0.1 | 412.3 ± 42.7 |
| board latency [ms/image] | **13.62 ± 0.00** | 57.13 ± 0.10 |
| board latency min / max [ms] | 5.5 / 25.6 | 23.4 / 79.1 |
| board cycles/image | 1 362 208 | 5 712 866 ± 10 229 |
| cycles per executed op | 67.5 | 53.5 ± 0.1 |
| model size in BRAM [B] | 103 024 | 103 024 |

The PC reference from `benchmark.py` (`runs/summary.md`) is: ANN 96.82 ± 0.18 %,
SNN 97.64 ± 0.07 %, ANN loss 0.1091, SNN loss 0.0776. The board reproduces
it.

### 4.2 Latency distribution [ms/image] (optimised)

| model | mean | p5 | median | p95 | min | max |
|---|---|---|---|---|---|---|
| ANN (seed 0) | 13.62 | 8.32 | 13.62 | 18.70 | 5.56 | 25.57 |
| SNN (seed 0) | 57.06 | 44.28 | 57.61 | 66.68 | 22.96 | 78.50 |

Both latencies depend on the data: images with more ink cost more. The
per-image latencies of the two models correlate with r = 0.83.

### 4.3 Effect of the optimisations (first 1000 images, same images)

| model | baseline [ms] | optimised [ms] | speed-up | per-image range |
|---|---|---|---|---|
| ANN | 135.79 (constant: dense) | 12.96 | **10.5×** | 5.7× – 24× |
| SNN | 402.7 | 56.5 | **7.1×** | 3.1× – 15.5× |

At **baseline level the ANN is already 3.0× faster** (135.8 vs 402.7 ms).
Both baselines cost the same per operation, 133.6 vs 129.6 cycles. The SNN
simply performs 3× more operations (310 k SOPs vs 101.6 k MACs). The
optimisations widen the gap to 4.2×, because input sparsity helps the ANN
more than delta coding helps the SNN.

### 4.4 Where the cycles go (cost model, all 10 000 images, R² ≥ 0.995)

```
ANN:  cycles ≈ 7 571 · (nonzero pixels) + 1 198 · (nonzero hidden) + 0.12 M
SNN:  cycles ≈ 3 450 · (input changes)  +   124 · (hidden spikes)  + 2.89 M
```

Breakdown of the mean latency (seed 0):

| part | ANN | SNN |
|---|---|---|
| input layer | 151.1 × 7 571 = 1.14 M cycles = **11.4 ms (84 %)** | 802.4 × 3 450 = 2.77 M = **27.7 ms (49 %)** |
| hidden → output layer | 82.5 × 1 198 = 0.10 M = 1.0 ms (7 %) | 382 × 124 = 0.05 M = 0.5 ms (1 %) |
| fixed per image | 0.12 M = 1.2 ms (9 %) | **2.89 M = 28.9 ms (51 %)** |
| **total** | **13.6 ms** | **57.1 ms** |

Reading this:

- **Per synaptic operation the SNN is cheaper:** 27 cycles per add vs 59
  cycles per MAC (128 synapses per row). An SNN synapse is an add, an ANN
  synapse needs a multiply too. That is the classic SNN argument, and it
  holds here.
- **But the SNN does 5.3× more row operations** (802 input changes vs 151
  nonzero pixels). Rate coding spreads one pixel over 25 timesteps, and grey
  pixels toggle.
- **And the SNN pays ~1.16 ms per timestep regardless of input**: the LIF
  update of 138 neurons (64-bit multiply for the Q16 leak), lane flushes and
  bitmap bookkeeping. Over T = 25 that is half its latency. The ANN does the
  equivalent work (ReLU + requantization) once.
- 43.5 ms of the 4.2× gap ≈ 16.3 ms more input-layer work + 27.7 ms more
  fixed per-timestep work.

### 4.5 Paired per-image view (same test image, optimised)

| | seed 0 | seed 1 |
|---|---|---|
| SNN / ANN latency, mean | 4.19 | 4.20 |
| median (min – max) | 4.27 (2.23 – 7.72) | 4.27 (2.31 – 7.75) |
| images where the SNN is faster | **0 / 10 000** | **0 / 10 000** |
| both correct | 9 637 | 9 617 |
| only ANN correct | 58 | 56 |
| only SNN correct | 135 | 140 |
| both wrong | 170 | 187 |
| same prediction | 9 775 | 9 768 |

Per digit (mean over seeds): the SNN is more accurate for every class. The
digit "1" (least ink) is the cheapest for both: ANN 8.8 ms, SNN 45.0 ms.
"0" is the most expensive: ANN 16.5 ms, SNN 62.9 ms. Full table in
`results/ann_vs_snn/comparison_tables.md`.

### 4.6 Energy

There is no power measurement yet. PicoRV32 runs at a fixed clock with no
clock gating, and the PS and fabric draw roughly constant power during a
run. **Energy per inference on this platform is therefore ≈ proportional to
cycles**, so the ANN uses about 4.2× less energy per image. This is an
inference from the cycle counts, not a measurement. Measuring board power
(e.g. a USB power meter, or the PMBus/INA rails if available) during
`benchmark_pynq.py` would close this gap.

### 4.7 Memory

| | ANN | SNN |
|---|---|---|
| model (weights + biases + header) | 103 024 B | 103 024 B |
| input per image | 784 B | 2 500 B (25 × 100 B bitmaps) |
| neuron state that persists over time | none (one pass) | membrane + current per neuron (138 × 8 B used) |

---

## 5. Interpretation for the research question

> *For edge inference on a resource-constrained processor, is a spiking
> network a better design choice than a classical ANN, and what hardware
> support is worth adding to make it win?*

On a **general-purpose, in-order, scalar RISC-V core** with this network
and rate coding with T = 25, the answer from this first measurement is:
**no on latency/energy (4.2× slower), yes on accuracy (+0.8 pp).**

The SNN's per-synapse cost advantage (add vs multiply, 27 vs 59 cycles) is
real but outweighed by:

1. **Temporal unrolling.** T = 25 timesteps multiply the work. Even with
   exact delta coding, the SNN processes 5.3× more rows than the sparse ANN.
2. **Per-timestep neuron overhead.** This is half the SNN latency, and it
   is exactly what a CPU does badly and dedicated hardware does well.
3. **Input sparsity is not unique to SNNs.** MNIST is 81 % zeros, and a
   sparse ANN exploits that just as well, in one pass.

What would shift the balance (next experiments, all measurable with the
current tooling):

- **Fewer timesteps.** SNN cost is ≈ 2.3 ms per timestep. Linear
  extrapolation puts break-even with the ANN at **T ≈ 6**. Whether accuracy
  survives T ≈ 6 needs retraining with `--timesteps 6` and a re-run. This
  is the most direct lever and needs no hardware change.
- **A different encoding** (latency / time-to-first-spike, or delta
  encoding of the input) would reduce input events far more than rate
  coding.
- **ISA extension (the project's extra part)** aimed at the SNN's fixed
  cost: a LIF instruction (leak multiply + add + compare + reset in one
  instruction) targets the 51 % fixed part. A packed "4×int8 add into lanes"
  targets the input layer. Note that a packed MAC instruction would speed up
  the ANN too, so measure both.
- **Bus wait states** (`READ_WAIT = 8` in `rtl/spike_soc.v`) slow both models
  equally per instruction. Changing them wouldn't change the ratio much, but
  would change absolute numbers.

---

## 6. Caveats

- One network size (784-128-10), one dataset, one encoding (rate, T = 25),
  two seeds. The std across seeds is small (latency std ≤ 0.2 %), but two
  seeds is the minimum for an error bar.
- The "same optimisation effort" is a judgement. Each kernel got the same
  techniques (sparsity, SWAR loads, register blocking) and was checked for
  spills, but neither is the best possible. The baseline numbers are
  reported too, so both levels can be compared.
- Energy is inferred from cycles, not measured (§4.6).
- The ANN activation scale is calibrated on training data (max). A clipped
  percentile might gain accuracy, but quantization already costs nothing
  here.

---

## 7. Reproduce

From `code/ann_vs_snn_mnist/` in Git Bash, with the venv active and the
board reachable:

```bash
pynq/build_fw.sh                                     # all firmware images
python pynq/benchmark_pynq.py                        # optimised, full test set (~25 min)
python pynq/benchmark_pynq.py --fw baseline --n 1000 # baselines (~18 min)
python pynq/compare_pynq.py --out compare.md         # the breakdown tables of §4
```

`benchmark_pynq.py` exports any missing model (`export_model.py`), runs it
on the board (`deploy_pynq.sh`) and writes `runs/summary_pynq_<fw>.md/json`.

## 8. Raw data (`docs/results/ann_vs_snn/`)

| file | content |
|---|---|
| `{ann,snn}_seed{0,1}_fast_board_results.csv` | one row per test image (10 000): label, board / PC-int / PC-float prediction, `outputs_match`, cycles, input / hidden / output events, input changes, host ms, outputs |
| `{ann,snn}_seed{0,1}_baseline_board_results.csv` | same, baseline firmware, first 1000 images |
| `{ann,snn}_seed{0,1}_{fast,baseline}_board_summary.json` | per-run summary from the board |
| `{ann,snn}_seed{0,1}_meta.json` | quantization scales, thresholds / requant multipliers, PC accuracies |
| `summary_pynq_{fast,baseline}.{md,json}` | the benchmark tables (mean ± std over seeds) |
| `summary_pc_benchmark.md` | `benchmark.py` on the PC, for reference |
| `comparison_tables.md` | output of `compare_pynq.py` (`--results docs/results/ann_vs_snn`; §6–8: muRISCV-NN) |
| `ann_seed{0,1}_muriscv_*`, `summary_pynq_muriscv.*` | the muRISCV-NN reference level, see [muriscv_reference.md](muriscv_reference.md) |
