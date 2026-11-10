# ANN vs SNN on the PYNQ-Z2 RISC-V core: MIT-BIH ECG

*Measured 2026-10-02. ECG counterpart of
[ann_vs_snn_pynq_comparison.md](ann_vs_snn_pynq_comparison.md) (MNIST).*

Two networks from `code/ann_vs_snn_ecg/train_ecg.py` classify heartbeats of
the MIT-BIH Arrhythmia Database into the AAMI classes N / S / V, on patients
they have never seen. Both run on the same PicoRV32 soft core, with the same
unchanged bitstream, the **same firmware image as MNIST**, the same int8
quantisation scheme and the same two levels of software optimisation.

---

## 1. Headline

| (mean ± std over seeds 0, 1, 2; all 49 273 DS2 beats) | ANN | SNN |
|---|---|---|
| board macro-F1 (N / S / V) | 0.675 ± 0.004 | **0.706 ± 0.007** |
| board accuracy | 93.62 ± 0.12 % | **94.18 ± 0.11 %** |
| Se / +P, S beats | 17.3 / 29.7 % | **23.4 / 38.7 %** |
| Se / +P, V beats | 93.3 / 76.1 % | **94.9 / 78.0 %** |
| board latency, optimised firmware | **4.87 ms** | 8.30 ± 0.04 ms |
| board latency, baseline firmware | **11.70 ms** | 13.46 ± 0.02 ms |
| ops/beat as defined in `benchmark_ecg.py` | 8 448 MACs | 8 256 MACs + 194 SOPs |
| bit-exact vs PC int8 reference | 147 819 / 147 819 | 147 819 / 147 819 |
| (same, baseline firmware) | 147 819 / 147 819 | 147 819 / 147 819 |

**On this platform the ANN is 1.70× faster than the SNN, on every single
test beat. On DS2 the SNN is more accurate (+0.031 macro-F1), mainly on
supraventricular (S) beats, but in the cross-validation on DS1 it was the
other way round (§5), so this accuracy edge is not a robust result.** The
gap is much smaller than on MNIST (4.2×): with direct encoding the SNN's
first layer is exactly the ANN's first layer, so the whole difference is
the cost of simulating T = 10 timesteps, which a measured latency-vs-T
sweep puts at 0.38 ms per step (§6).

**External reference (added 2026-10-07):** the same int8 ANN on the
unmodified muRISCV-NN library (RISC-V CMSIS-NN, scalar C) takes 7.05 ms per
beat, bit-exact on the board. That is faster than our fast SNN (8.30 ms) but
slower than our fast ANN (4.87 ms) and the SNN coprocessor (5.32 ms). On
MNIST the same library loses to the fast SNN; on the dense ECG inputs it
does not. See [muriscv_reference.md](muriscv_reference.md) §7.

---

## 2. What was compared, and how fair it is

| held equal | how |
|---|---|
| network | 129-64-3, 8 515 parameters, same training budget (`train_ecg.py`: AdamW, lr 1e-3, wd 0.2, cosine schedule, 10 epochs, batch 64, last epoch kept, class-weighted CE, same augmentation stream, same init seed and data order) |
| data protocol | de Chazal inter-patient split; all tuning by patient-level 5-fold CV on DS1; DS2 (22 unseen patients) used once for the final models |
| test data | the same 49 273 DS2 beats in the same order, for both models and all seeds (paired per-beat results) |
| input | the **same int8 bytes** for both models (129 features: beat − causal patient template, 3 RR features), one input scale calibrated on DS1 only |
| hardware | PicoRV32 RV32IM @ 100 MHz, unchanged `spike_top.bit`, 256 KB BRAM |
| software | **one** firmware image (v3.1) runs ANN and SNN, MNIST and ECG (type from the `model.bin` magic); same compiler and flags |
| weights | int8, symmetric, one scale per layer, int32 biases, same 9 620 B model layout |
| correctness | each board output (ANN logits / SNN spike counts) equals the PC int8 model **bit for bit**, for every beat, seed and firmware level |
| optimisation effort | two builds: a straightforward baseline and the same set of optimisations for both (§3) |

**What necessarily differs:**

- **Encoding.** The SNN uses *direct encoding*: the feature vector is a
  constant input current. Its first layer is one int8 MAC pass,
  `I = b + W x`, computed once and reused for all T = 10 steps; the hidden
  and output layers are LIF neurons and spike-driven. (Rate encoding was
  also tried in the CV and was worse, see §5.)
- **Hidden activations.** SNN: binary spikes, no requantisation. ANN: uint8
  via `min(255, (relu(acc)·m_q + 2²³) >> 24)`, scale calibrated on the
  (quantised) DS1 training features.
- **Readout.** SNN: argmax of the output spike counts. ANN: argmax of the
  int32 logits. Both take the first maximum on ties, on the PC and on the
  board.

**Tuning effort.** The shared choices (input morphology, augmentation,
epochs, weight decay) were tuned on the ANN (faster to train) and then
applied unchanged to the SNN; the SNN got its own small sweep over T, β and
encoding (10 configurations). If anything this favours the ANN. All CV
results: `docs/results/ann_vs_snn_ecg/pc/cv/ALL.md`.

### 2.1 Input quantisation

Unlike MNIST pixels, the ECG features are real-valued and signed, so the
board gets `x_q = clip(round(x / s_x), −127, 127)`, int8, with one
symmetric scale for all 129 features. The maximum cannot set the scale: one
RR ratio in DS1 reaches 294 while the median |x| is 0.08, which would map
97 % of the inputs to 0. `s_x` is therefore set by a clip percentile of |x|
over the **DS1** features, chosen by the float/int8 prediction agreement on
DS1 (summed over the six models):

| clip at | s_x | zero inputs (DS1) | DS1 float/int8 agreement, mean of 6 models |
|---|---|---|---|
| P99.9 | 0.0347 | 14.1 % | 99.77 % |
| P99.95 | 0.0451 | 18.1 % | 99.80 % |
| **P99.99** | **0.0578** | 22.8 % | **99.80 %** |
| P99.999 | 0.0725 | 27.9 % | 99.75 % |

Cost of the port, float → int8, measured once on all of DS2 (PC):

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| ANN macro-F1 | 0.670 → 0.670 | 0.677 → 0.677 | 0.678 → 0.678 |
| SNN macro-F1 | 0.703 → 0.702 | 0.702 → 0.702 | 0.713 → 0.714 |
| ANN / SNN accuracy | 93.52 → 93.48 / 94.10 → 94.07 % | 93.69 → 93.71 / 94.21 → 94.19 % | 93.64 → 93.66 / 94.30 → 94.29 % |
| float/int8 prediction agreement, ANN / SNN | 99.84 / 99.64 % | 99.82 / 99.57 % | 99.84 / 99.57 % |

Quantisation costs at most 0.001 macro-F1. The accuracy difference between
the models comes from training, not from the port. (The SNN thresholds in
integer units are 6 971 for the hidden layer, whose current has scale
s_w·s_x, and 625 for the output layer.)

---

## 3. The two firmware levels

The firmware is the MNIST one, extended (v3.1) with two model types: `ANN2`
(int8 inputs) and `SNN3` (int8 inputs, direct encoding). MNIST still runs
on it bit-exactly (§7).

| | ANN | SNN (direct) |
|---|---|---|
| **baseline** | dense textbook layers: every input (zeros included) × every output, `lb + mul + add`, accumulator in memory | first layer: the **same** dense MAC routine, once; then per timestep: LIF update of all neurons, and per hidden spike one weight row added byte by byte |
| **fast: sparsity** | skip zero inputs / zero ReLU outputs (103.7 of 129 inputs, 38.5 of 64 hidden are nonzero) | first layer: the same sparse routine as the ANN; hidden → output: **delta currents** (only hidden neurons whose spike state changed) |
| **fast: SWAR** | 1 load = 4 int8 weights (offset binary), even/odd bytes → two 16-bit lanes, one `mul` scales two weights; negative inputs in a second list whose products are subtracted | first layer: same; later layers: 4 int8 weights per load added into 16-bit lanes |
| **fast: blocking** | 8 outputs per pass in registers | first layer: 8 per pass; later layers: 16 outputs in 8 registers |

The first layer of both models is literally the same function
(`mac_layer()` in `firmware/ann.c`), so the comparison isolates the cost of
the SNN's temporal part. Both inner loops of `mac_layer` (positive and
negative inputs) were checked in the disassembly: 33 instructions per input
row for 8 outputs, no stack accesses.

---

## 4. Results

### 4.1 Benchmark table (optimised firmware, all DS2 beats)

Output of `python pynq/benchmark_pynq_ecg.py` (`summary_pynq_fast.md`):

| metric (mean ± std over seeds) | ANN | SNN |
|---|---|---|
| board macro-F1 (N/S/V) | 0.675 ± 0.004 | 0.706 ± 0.007 |
| board accuracy [%] | 93.62 ± 0.12 | 94.18 ± 0.11 |
| Se N / S / V [%] | 96.80 / 17.30 / 93.34 | 97.07 / 23.40 / 94.93 |
| +P N / S / V [%] | 96.73 / 29.72 / 76.08 | 97.01 / 38.67 / 78.01 |
| board loss (class-weighted CE) | 0.3044 ± 0.0063 | 0.3053 ± 0.0107 |
| PC float macro-F1, same beats | 0.675 ± 0.004 | 0.706 ± 0.006 |
| board == PC int8 reference [%] | 100.00 | 100.00 |
| parameters | 8 515 | 8 515 |
| ops/beat, `benchmark_ecg.py` | 8 448 | 8 450 ± 11 |
| ops/beat executed (fast) | 6 754 | 6 833 ± 11 |
| nonzero int8 inputs per beat (of 129) | 103.7 | 103.7 |
| nonzero hidden / hidden spikes per beat | 38.5 ± 1.1 | 64.6 ± 3.7 (over 10 steps) |
| output spikes per beat | – | 5.0 ± 0.3 |
| board latency [ms/beat] | **4.872 ± 0.009** | 8.300 ± 0.035 |
| board latency min / max [ms] | 0.82 / 5.94 | 3.64 / 10.17 |
| board cycles/beat | 487 248 | 829 997 ± 3 465 |
| model size in BRAM [B] | 9 620 | 9 620 |

The PC reference (`benchmark_ecg.py`, `pc/summary_pc_benchmark_ds2.md`):
ANN 0.675 ± 0.004 / 93.62 %, SNN 0.706 ± 0.006 / 94.20 %, loss 0.3045 /
0.3051. The board reproduces it.

Confusion matrices on the board (rows = true N/S/V, columns = predicted,
summed over the 3 seeds):

```
ANN    128 414   1 928   2 312        SNN    128 761   1 770   2 123
         4 032     953     523                 3 758   1 289     461
           315     328   9 014                   215     275   9 167
```

### 4.2 Latency distribution [ms/beat] (optimised, all beats)

| model | seed | mean | p5 | median | p95 | min | max |
|---|---|---|---|---|---|---|---|
| ANN | 0 | 4.88 | 3.75 | 4.96 | 5.70 | 0.85 | 5.96 |
| ANN | 1 | 4.87 | 3.74 | 4.95 | 5.70 | 0.81 | 5.93 |
| ANN | 2 | 4.87 | 3.73 | 4.95 | 5.70 | 0.80 | 5.92 |
| SNN | 0 | 8.34 | 6.93 | 8.36 | 9.55 | 3.64 | 10.29 |
| SNN | 1 | 8.28 | 6.83 | 8.31 | 9.51 | 3.57 | 10.06 |
| SNN | 2 | 8.28 | 6.87 | 8.29 | 9.51 | 3.70 | 10.14 |

The spread is much narrower than on MNIST: the int8 ECG features are
mostly nonzero, so the input layer does almost the same work for every
beat. The per-beat latencies of the two models correlate with r = 0.96.

### 4.3 Effect of the optimisations (all beats, same beats)

| model | baseline [ms] | optimised [ms] | speed-up | per-beat range |
|---|---|---|---|---|
| ANN | 11.70 (dense, constant) | 4.87 | **2.40×** | 2.0× – 14.6× |
| SNN | 13.46 | 8.30 | **1.62×** | 1.4× – 3.7× |

At **baseline level the ANN is 1.15× faster** (11.70 vs 13.46 ms). The
baseline ANN costs 138.5 cycles per MAC (MNIST: 133.6). The optimisations
widen the gap to 1.70×, because only 20 % of the inputs are zero (MNIST:
81 %), so sparsity helps less here, and SWAR speeds up the MAC-heavy first
layer (shared by both models) more than the SNN's per-timestep neuron
updates.

### 4.4 Where the cycles go

Cost model of the fast kernels (least squares over all beats):

```
ANN:  cycles ≈ 3 920 · (nonzero inputs) + 820 · (nonzero hidden) + 49 k    R² = 0.9999
SNN:  cycles ≈ 4 470 · (nonzero inputs) + 250–310 · (hidden spikes) + 345 k  R² = 0.94–0.96
```

The SNN fit is weaker because the hidden-spike count and the input count
are correlated. A cleaner decomposition comes from running the same trained
SNN (seed 0) for different T on the board, each run bit-exact against its
own int8 reference (first 2 000 DS2 beats, fast firmware,
`results/ann_vs_snn_ecg/T_study/`):

| T | 1 | 2 | 5 | **10** | 20 |
|---|---|---|---|---|---|
| SNN latency [ms/beat] | 4.44 | 4.73 | 5.84 | **7.77** | 11.64 |
| SNN / ANN (ANN: 4.44 ms on these beats) | 1.00 | 1.07 | 1.31 | **1.75** | 2.62 |
| board == PC int8 | 2000/2000 | 2000/2000 | 2000/2000 | 2000/2000 | 2000/2000 |
| macro-F1 on DS2 (int8, model trained at T = 10) | 0.339 | 0.635 | 0.687 | 0.702 | 0.695 |

```
SNN latency = 3.99 ms + 0.381 ms · T     (R² = 0.9997)
```

Breakdown of the mean latency on those beats:

| part | ANN | SNN (T = 10) |
|---|---|---|
| first layer: input → hidden current | ≈ 3.6 ms (one sparse MAC pass) | ≈ 3.6 ms (the **same** pass) |
| hidden → output layer, requant / neuron state | ≈ 0.3 ms (38.5 rows × 820 cycles) | 10 × 0.381 ms = **3.81 ms (49 %)** |
| fixed per beat (input copy, argmax, setup) | ≈ 0.5 ms | ≈ 0.4 ms |
| **total** | **4.44 ms** | **7.77 ms** |

Reading this:

- **The SNN's first layer costs exactly what the ANN's costs**, because it
  is the same computation (direct encoding). There is no event-driven saving
  in the first layer: the input is analog and constant.
- **Each timestep costs 0.381 ms** (38 k cycles): the LIF update of
  67 neurons (Q16 leak multiply, threshold, reset), the hidden → output
  delta update and the spike bookkeeping. The ANN does the equivalent work
  (ReLU, requantisation and a 64 × 3 layer) once, in ≈ 0.3 ms.
- Per synaptic operation the SNN is again cheaper: a hidden → output
  update is an add, 250–310 cycles per hidden spike (3 synapses) vs
  820 cycles per nonzero hidden activation in the ANN. But there are only
  194 such operations per beat; they are not where the time goes.

### 4.5 Paired per-beat view (same DS2 beat, optimised)

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| SNN / ANN latency, mean | 1.71 | 1.70 | 1.70 |
| median (min – max) | 1.70 (1.52 – 4.52) | 1.69 (1.51 – 4.53) | 1.69 (1.52 – 4.65) |
| beats where the SNN is faster | **0 / 49 273** | **0 / 49 273** | **0 / 49 273** |
| both correct | 45 744 | 45 853 | 45 858 |
| only ANN correct | 316 | 321 | 289 |
| only SNN correct | 605 | 556 | 601 |
| both wrong | 2 608 | 2 543 | 2 525 |
| same prediction | 48 225 | 48 264 | 48 256 |

### 4.6 Per class and per record

| class | ANN Se [%] | SNN Se [%] | ANN latency [ms] | SNN latency [ms] |
|---|---|---|---|---|
| N (44 218 beats) | 96.80 | 97.07 | 4.82 | 8.22 |
| S (1 836) | 17.30 | 23.40 | 5.17 | 8.69 |
| V (3 219) | 93.34 | 94.93 | 5.48 | 9.18 |

Abnormal beats cost a little more for both models: their beat − template
difference has more nonzero int8 samples (mean nonzero inputs: N 102,
S 111, V 121 of 129).

Per record (board, mean over seeds; Se is "-" where the record has no beat of
that class):

| record | N / S / V beats | ANN acc | SNN acc | ANN Se S | SNN Se S | ANN Se V | SNN Se V |
|---|---|---|---|---|---|---|---|
| 100 | 2237 / 33 / 1 | 0.995 | 0.992 | 0.72 | 0.45 | 1.00 | 1.00 |
| 103 | 2080 / 2 / 0 | 0.997 | 0.998 | 0.50 | 0.33 | - | - |
| 105 | 2524 / 0 / 41 | 0.930 | 0.944 | - | - | 0.29 | 0.29 |
| 111 | 2121 / 0 / 1 | 0.956 | 0.963 | - | - | 1.00 | 1.00 |
| 113 | 1787 / 6 / 0 | 1.000 | 0.997 | 1.00 | 0.11 | - | - |
| 117 | 1532 / 1 / 0 | 1.000 | 1.000 | 1.00 | 1.00 | - | - |
| 121 | 1859 / 1 / 1 | 0.994 | 0.994 | 0.67 | 1.00 | 1.00 | 1.00 |
| 123 | 1513 / 0 / 3 | 0.998 | 0.998 | - | - | 1.00 | 1.00 |
| 200 | 1742 / 30 / 825 | 0.975 | 0.976 | 0.23 | 0.16 | 0.97 | 0.97 |
| 202 | 2059 / 55 / 19 | 0.917 | 0.932 | 0.76 | 0.75 | 1.00 | 0.95 |
| 210 | 2421 / 22 / 195 | 0.975 | 0.980 | 0.21 | 0.14 | 0.92 | 0.92 |
| 212 | 2746 / 0 / 0 | 0.872 | 0.871 | - | - | - | - |
| 213 | 2639 / 28 / 220 | 0.975 | 0.982 | 0.01 | 0.04 | 0.83 | 0.91 |
| 214 | 2001 / 0 / 256 | 0.997 | 0.994 | - | - | 0.98 | 0.95 |
| 219 | 2080 / 7 / 64 | 0.970 | 0.968 | 0.29 | 0.24 | 0.81 | 0.81 |
| 221 | 2029 / 0 / 396 | 0.996 | 0.995 | - | - | 0.99 | 0.99 |
| 222 | 2272 / 209 / 0 | 0.804 | 0.807 | 0.71 | 0.72 | - | - |
| 228 | 1686 / 3 / 362 | 0.991 | 0.992 | 1.00 | 0.78 | 1.00 | 1.00 |
| 231 | 1566 / 1 / 2 | 0.958 | 0.969 | 0.00 | 0.00 | 1.00 | 1.00 |
| **232** | 397 / **1381** / 0 | 0.262 | 0.334 | 0.05 | 0.14 | - | - |
| 233 | 2229 / 7 / 830 | 0.967 | 0.978 | 0.62 | 0.57 | 0.89 | 0.94 |
| 234 | 2698 / 50 / 3 | 0.983 | 0.984 | 0.09 | 0.14 | 1.00 | 1.00 |

Record 232 holds 75 % of all S beats in DS2, and both models miss most of
them. The SNN's higher S sensitivity comes almost entirely from this one
record (5 % → 14 %). Record 212 (right bundle-branch block, all N) and
222 (atrial flutter / fibrillation episodes) are the main sources of false
alarms for both.

### 4.7 Energy

There is no power measurement. PicoRV32 runs at a fixed clock with no clock
gating, and the board's power is roughly constant during a run, so
**energy per beat on this platform is ≈ proportional to cycles**: the ANN
uses ≈ 1.7× less energy per beat. This is an inference from cycle counts,
not a measurement. In a real monitor a beat arrives about once per second,
so both models keep the core busy < 1 % of the time (0.5 vs 0.8 %); the
difference only matters if the core can sleep in between, which the current
firmware does not do (it busy-polls the mailbox).

### 4.8 Memory

| | ANN | SNN |
|---|---|---|
| model (weights + biases + header) | 9 620 B | 9 620 B |
| input per beat | 132 B (129 int8, padded) | 132 B (same bytes) |
| state that persists over time | none (one pass) | membrane + current for 67 neurons (8 B each) |
| firmware (fast image, both models, MNIST + ECG) | 63 072 of 65 536 B code + data | (same image) |

---

## 5. Interpretation for the research question

> *For edge inference on a resource-constrained processor, is a spiking
> network a better design choice than a classical ANN, and what hardware
> support is worth adding to make it win?*

For ECG beat classification on a general-purpose, in-order, scalar RISC-V
core, the answer is: **no on latency and energy (1.70× slower on every
beat), and not clearly on accuracy.**

- **Accuracy.** On DS2 the SNN is better in all three seeds (+0.031
  macro-F1, +6 pp S sensitivity, +1.6 pp V sensitivity). In the
  cross-validation on DS1, which is what the configuration was chosen on,
  the ANN was better (0.672 ± 0.001 vs 0.655 ± 0.001). The SNN's DS2
  advantage comes largely from one record (232). So it is better read as
  "the SNN is as accurate as the ANN" than as a win.
- **Latency.** With direct encoding the SNN's first layer *is* the ANN's
  first layer. Everything the SNN adds is temporal: 0.38 ms per timestep,
  against ≈ 0.3 ms for the ANN's entire second stage. The measured
  break-even is **T ≈ 1.2**, where the SNN is effectively a one-step
  thresholded ANN and its accuracy collapses (macro-F1 0.34 at T = 1 for a
  model trained at T = 10). In the CV a model *trained* at T = 5 reached
  0.652, at a measured 1.31× the ANN latency.

What would shift the balance:

- **A LIF instruction / neuron-update unit.** The per-timestep work (Q16
  leak multiply, add, compare, reset for 67 neurons plus bookkeeping) is
  49 % of the SNN latency and the only part the ANN does not have. A
  custom instruction that does one LIF update per cycle would cut it from
  ≈ 570 cycles per neuron-step to a few tens; that would put the SNN at
  roughly the ANN's latency for T = 10, but not below it.
- **Fewer timesteps, trained for.** A model trained at T = 5 lost only
  0.003 macro-F1 in the CV (0.652 vs 0.655) and would save 1.9 ms per beat.
- **Encodings that make the first layer sparse.** Direct encoding gives the
  SNN no input sparsity at all. The real case for an SNN on ECG is the
  streaming setting (one sample at a time, delta / level-crossing
  encoding), where most samples of a beat cause no input event; there, the
  fair ANN counterpart is a recurrent network that also has to run every
  sample. That is the stretch goal and not measured here.

---

## 6. ECG vs MNIST: why the ratio drops from 4.2× to 1.7×

| | MNIST (784-128-10) | ECG (129-64-3) |
|---|---|---|
| SNN encoding | rate code, T = 25 | direct, T = 10 |
| inputs that are zero | 81 % (pixels) | 20 % (int8 features) |
| ANN input rows processed per sample | 151 of 784 | 103.7 of 129 |
| SNN input rows processed per sample | 802 input changes (over 25 steps) | 103.7 (once, same as the ANN) |
| SNN input-layer work / ANN input-layer work | **5.3×** | **1.0×** |
| SNN cost per timestep, beyond the input layer | ≈ 1.16 ms (138 neurons) | 0.381 ms (67 neurons, all included) |
| per-timestep part of the SNN latency | 51 % | 49 % |
| SNN / ANN latency, optimised | **4.2×** | **1.70×** |
| SNN / ANN latency, baseline | 3.0× | 1.15× |
| break-even T | ≈ 6 (extrapolated) | **≈ 1.2 (measured)** |

1. **Input sparsity.** MNIST is 81 % zero pixels, and a sparse ANN skips
   them in one pass. That made the ANN's input layer cheap and the SNN's
   rate-coded input layer 5.3× more expensive. ECG features are dense, so
   sparsity buys the ANN little (only 2.4× from all optimisations, vs 10.5×
   on MNIST), and the ANN's advantage in the first layer disappears.
2. **Encoding.** Rate coding makes the SNN process input events every
   timestep (802 row updates per image even with exact delta coding).
   Direct encoding processes the input once, exactly like the ANN. This
   removes the largest of the two MNIST penalties.
3. **T and network size.** The remaining penalty, the neuron state that
   must be updated every step, scales with T × (hidden + output neurons):
   25 × 138 on MNIST, 10 × 67 on ECG. That is why both the absolute
   per-step cost and the ratio are smaller.
4. **Break-even.** On MNIST the SNN would have needed T ≈ 6 to break even
   (the input layer's event count also shrinks with T). On ECG the first
   layer is identical, so the SNN only breaks even when its T timesteps
   cost no more than the ANN's second stage, i.e. at T ≈ 1. **With direct
   encoding an SNN cannot be faster than the equivalent ANN on a scalar
   CPU; the best it can do is tie.** An SNN beats an ANN here only if its
   input is genuinely event-driven and sparse, *and* the per-timestep
   neuron update is cheap (hardware support), not just one of the two.

---

## 7. Caveats

- **Preprocessing runs on the host**: band-pass filter, decimation,
  R-peak positions (from the annotations), per-beat z-score, RR features
  and the causal template (median of the previous 32 beats). It is
  identical for both models and not in the measured latency. On the core,
  the template alone (a 32-beat sample-wise median per beat) would cost
  more than either network.
- **The SNN's DS2 accuracy advantage is not supported by the CV** (§5) and
  is concentrated in one record. Inter-patient S detection is poor for both
  models (Se 17–23 %), as is common in the literature for this protocol.
- **The T sweep (§4.4) is a latency study.** It reuses the model trained at
  T = 10; its DS2 accuracies at other T are a post-hoc sensitivity, not a
  tuning step (T = 10 was fixed by the CV beforehand).
- One network size, one SNN encoding on the board (direct). Rate encoding
  was worse in the CV (0.637–0.639 vs 0.640–0.655) and its cost on the
  board is covered by the MNIST measurements.
- The "same optimisation effort" is a judgement: each kernel got the same
  techniques (sparsity / delta, SWAR, register blocking). The baseline
  numbers are reported too.
- Energy is inferred from cycles, not measured (§4.7).
- The loss in the tables is `benchmark_ecg.py`'s definition (class-weighted
  CE averaged per batch of 1 024 beats); a single global weighted mean
  gives ≈ 0.47 instead of ≈ 0.30 for both models.
- Firmware v3.1 replaces v3.0 for MNIST too. MNIST outputs are unchanged
  (200 images per model and level, all identical to v3.0 and to the PC int8
  reference); cycle counts changed by +0.1 % to +1.1 %.
- The fast firmware image has 2.4 KB of its 64 KB code/data region left.

---

## 8. Reproduce

From `code/ann_vs_snn_ecg/` in Git Bash, venv active, board reachable:

```bash
python train_ecg.py --final --use-rr --morph diff --augment --epochs 10 \
    --weight-decay 0.2 --timesteps 10 --beta 0.9 --encoding direct
python benchmark_ecg.py                                 # PC reference on DS2
python pynq/export_ecg.py                               # int8 export, all DS2 beats
python pynq/benchmark_pynq_ecg.py                       # optimised firmware (~45 min)
python pynq/benchmark_pynq_ecg.py --fw baseline         # baseline (~75 min)
python ../ann_vs_snn_mnist/pynq/compare_pynq.py --task ecg \
    --export-root pynq/export --out runs/compare_pynq.md
# latency vs T (seed 0, first 2000 beats)
for T in 1 2 5 20; do
  python pynq/export_ecg.py --timesteps $T --n-samples 2000 --out pynq/export_T runs/snn_ecg_seed0.pt
  ../ann_vs_snn_mnist/pynq/deploy_pynq.sh 192.168.2.99 pynq/export_T/snn_ecg_seed0_T$T
done
```

For long runs, `deploy_pynq.sh <ip> <export-folder> --detach -- --fw <fw>`
runs on the board in the background and `--resume` continues after an
interruption (`status_pynq.sh` fetches the results).

## 9. Raw data (`docs/results/ann_vs_snn_ecg/`)

| file | content |
|---|---|
| `{ann,snn}_seed{0,1,2}_{fast,baseline}_board_results.csv.gz` | one row per DS2 beat (49 273): label, board / PC-int / PC-float prediction, `outputs_match`, cycles, nonzero inputs, hidden events, output spikes, host ms, outputs (gzip) |
| `{ann,snn}_seed{0,1,2}_{fast,baseline}_board_summary.json` | per-run summary from the board |
| `{ann,snn}_seed{0,1,2}_meta.json` | input scale and its DS1 calibration, layer scales, thresholds / requant multipliers, PC float and int8 metrics |
| `summary_pynq_{fast,baseline}.{md,json}` | benchmark tables (mean ± std over seeds), confusion matrices, per-record and paired tables |
| `comparison_tables.md` | output of `compare_pynq.py --task ecg --results docs/results/ann_vs_snn_ecg` (§6–8: muRISCV-NN) |
| `ann_seed{0,1,2}_muriscv_*`, `summary_pynq_muriscv.*` | the muRISCV-NN reference level, see [muriscv_reference.md](muriscv_reference.md) §7 |
| `T_study/` | SNN seed 0 at T = 1, 2, 5, 20 on the board (first 2 000 beats) |
| `pc/` | PC side: every CV variant (`cv/`, `cv/ALL.md`), the DS2 PC benchmark, the final manifest |
