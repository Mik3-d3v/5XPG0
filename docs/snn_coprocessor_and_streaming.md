# SNN hardware support and streaming ECG on the PYNQ-Z2 RISC-V core

*Measured 2026-10-02 and 03. Follows [ann_vs_snn_ecg_pynq_comparison.md](ann_vs_snn_ecg_pynq_comparison.md)
(ECG, unchanged bitstream) and [ann_vs_snn_pynq_comparison.md](ann_vs_snn_pynq_comparison.md) (MNIST).*

Three steps:

1. **Rebuild the bitstream with Vivado 2026.1** and check that it behaves
   exactly like the released one (§1).
2. **Custom instructions for the SNN**: a small neuron coprocessor on
   PicoRV32's PCPI interface, aimed at the costs measured in the earlier
   reports. The ANN code is not touched (§2).
3. **Streaming ECG**: the setting where an SNN can use its strengths. A
   recurrent SNN reads the ECG sample by sample through a level-crossing
   encoder, a GRU of the same hidden size reads the same samples (§3).

### Headline (all outputs bit-exact against the PC int8 models)

| ECG, latency per beat on the PicoRV32 @ 100 MHz | ANN side | SNN side | macro-F1 (DS2) ANN / SNN |
|---|---|---|---|
| feedforward, software (ANN vs SNN, T = 10) | **4.87 ms** | 8.30 ms | 0.675 / 0.706 |
| feedforward, SNN on the coprocessor | **4.87 ms** | 5.32 ms | 0.675 / 0.706 |
| streaming, software (GRU vs recurrent SNN) | 1 099 ms | **36.4 ms** | 0.667 / 0.667 (int8) |
| streaming, SNN on the coprocessor | 1 100 ms | **4.59 ms** | 0.667 / 0.667 (int8) |

The coprocessor costs +889 LUTs, +342 registers and +2 DSPs (timing met at
100 MHz) and makes an SNN timestep 4.2× (feedforward) to 8× (streaming)
cheaper. In the streaming setting the event-driven SNN is 30× faster than a
GRU of the same hidden size in plain software and 240× faster with the
coprocessor, at equal accuracy on the held-out patients.

---

## 1. The bitstream with Vivado 2026.1

The course flow was built with Vivado 2024.1. With 2026.1 (this PC; the
install has the Zynq-7000 parts) the unchanged `vivado/build.tcl` builds
without errors; the critical warnings are the read-only PS7 AXI clock
parameters of the board preset (the script's filter misses the `CONFIG.`
prefix, Vivado ignores them) and the negative DDR DQS skews of the vendor
preset.

| | stock design, 2026.1 |
|---|---|
| timing @ 100 MHz | met, WNS +0.632 ns |
| Slice LUTs / registers | 2 631 / 2 550 |
| DSPs / BRAM tiles | 6 / 64 |
| on-chip power (Vivado estimate) | 1.682 W (PS dominates) |

On the board it is indistinguishable from the released bitstream: MNIST
(200 images) and ECG (2 000 beats), ANN and SNN, baseline and fast
firmware: every output **and every cycle count** is identical to the runs on
`spike_top.bit` (`results/snn_coprocessor/vivado_reference/`).

---

## 2. Custom instructions: an SNN neuron coprocessor

### 2.1 Where the SNN time went

The earlier reports measured where the SNN loses on this core:

- the **per-timestep neuron update** (leak multiply, add, compare, reset,
  bookkeeping): ≈ 570 cycles per neuron per step, 49 % (ECG) and 51 %
  (MNIST) of the SNN latency;
- adding **weight rows** for input / spike changes: 4 int8 weights per
  load, but ~1.5 instructions per synapse, each instruction fetch through
  the 8-wait-state memory.

### 2.2 Design

`code/pynqz2_riscv_flow/rtl/snn_pcpi.v`, attached to PicoRV32's
co-processor interface (PCPI) when the SoC is built with the Verilog define
`SNN_COPROC` (`vivado/build_snn.tcl` -> `spike_top_snn.bit`). The stock
build is unchanged.

| funct7 | instruction | operands | effect | cycles |
|---|---|---|---|---|
| 0x00 | `snn.cfg` | slot, β (Q16), threshold | parameters of one layer (4 slots) | 1 |
| 0x01 | `snn.clr` | first, count | membrane and current = 0 | count |
| 0x02 | `snn.setcur` | neuron, value | current = value (bias, direct input) | 1 |
| 0x03 | `snn.acc4` | neuron, 4 packed int8 weights | 4 currents ± 4 weights (offset-binary aware) | 1 |
| 0x04 | `snn.step` | slot, first, count ≤ 32 | LIF update of up to 32 neurons, returns their spike bits | count + 4 |
| 0x05 | `snn.get` | neuron | read membrane / current (testing) | 1 |

All R-type, opcode `custom-0`. The coprocessor holds membrane potential and
input current of 256 neurons in LUT RAM (4 current banks, so `acc4` updates
4 neurons per cycle). `snn.step` is a 4-stage pipeline (read, leak multiply
on 2 DSPs, integrate, compare + write back), one neuron per cycle, and
implements exactly the integer LIF of the PC reference:
`v = ((β·mem) >>> 16) + cur − (mem > thr ? thr : 0)`, spike = `v > thr`.

**Verification.** A Python model of the instruction semantics generates a
random stream of 4 005 instructions; an xsim testbench drives them through
the PCPI handshake as the CPU does and checks all 1 650 returned values:
0 errors (`code/pynqz2_riscv_flow/sim/`). A deliberately mutated RTL
(`>=` instead of `>` in the spike test) is caught. On the board every output
of every model is compared with the PC int8 reference, as before.

**Cost** (`results/snn_coprocessor/vivado_snn/`):

| | stock | with coprocessor | added |
|---|---|---|---|
| Slice LUTs | 2 631 | 3 520 | +889 (397 as LUT RAM) |
| Slice registers | 2 550 | 2 892 | +342 |
| DSPs | 6 | 8 | +2 |
| timing @ 100 MHz | WNS +0.632 ns | WNS +0.447 ns | met |
| on-chip power (estimate) | 1.682 W | 1.695 W | +13 mW (coprocessor) |

### 2.3 Firmware

A third firmware level, `snn_mnist_hw.bin` (`--fw hw`): the fast firmware
with only the SNN kernel replaced. Per timestep it adds the weight rows of
the inputs that changed with `snn.acc4` (one instruction per 4 synapses) and
updates every layer with `snn.step` (one instruction per 32 neurons). The
ANN code is the same object as in the fast image, and the ANN takes exactly
the same number of cycles on the new bitstream (2 000 / 2 000 ECG beats
identical).

### 2.4 Results (bit-exact, every beat / image)

| | ANN, fast | SNN, fast | SNN, hw | SNN speed-up | SNN / ANN |
|---|---|---|---|---|---|
| MNIST (784-128-10, T = 25, 2 seeds × 10 000) | 13.62 ms | 57.13 ms | **31.87 ms** | 1.79× | 4.19 → **2.34** |
| ECG (129-64-3, T = 10, 3 seeds × 49 273) | 4.87 ms | 8.30 ms | **5.32 ms** | 1.56× | 1.70 → **1.09** |

Board outputs equal the PC int8 reference for all 147 819 ECG and 20 000
MNIST runs on the hw level; accuracy is unchanged (ECG macro-F1 0.706,
MNIST 97.65 %).

**Latency vs T** (ECG SNN, seed 0, first 2 000 DS2 beats, each T bit-exact
against its own int8 reference; ANN on the same beats: 4.44 ms):

| T | 1 | 2 | 5 | 10 | 20 | fit |
|---|---|---|---|---|---|---|
| fast [ms] | 4.44 | 4.73 | 5.84 | 7.77 | 11.64 | 3.99 + **0.381**·T |
| hw [ms] | 4.06 | 4.10 | 4.34 | 4.81 | 5.75 | 3.92 + **0.091**·T |
| hw / ANN | 0.91 | 0.92 | 0.98 | 1.08 | 1.30 | |

The coprocessor cuts the cost of a timestep 4.2× (0.381 → 0.091 ms, the
remaining part is mostly the row additions and the spike bookkeeping in
software) and moves the **break-even with the ANN from T ≈ 1.2 to
T ≈ 5.7**. An ECG SNN trained at T = 5 reached 0.652 macro-F1 in the
cross-validation (T = 10: 0.655), so with this hardware it is as accurate as
before *and* faster than the ANN. What it cannot change: with direct
encoding the first layer is the same MAC pass as the ANN's (3.9 ms of the
4.06 ms at T = 1), so the SNN can only be ahead by the ANN's second stage.

**Fairness.** The coprocessor only helps the SNN, as intended. A packed
int8 multiply-accumulate instruction would speed up the ANN *and* the SNN's
direct-encoded first layer, which is exactly the part they share; that was
deliberately left out of scope here.

---

## 3. Streaming ECG: recurrent SNN vs GRU

### 3.1 Why a second ECG comparison

In the feedforward comparison the SNN's first layer is the ANN's first
layer (direct encoding), so the SNN can at best tie. An SNN has a real
structural advantage only when its input is a sparse event stream. An ECG
*is* a stream: a monitor sees one sample at a time, and for a normal beat
the signal hardly differs from the patient's recent beats. The fair ANN
counterpart in that setting is a recurrent network that also reads one
sample per step: a GRU (the model the keyword-spotting group uses too).

### 3.2 Models and protocol

Both models read the same stream, `d_t = beat − causal patient template`,
126 samples at 180 Hz, and get the same 3 RR features at the readout.

| | GRU | recurrent SNN (RSNN) |
|---|---|---|
| input per step | `d_t` (int8 on the board) | level-crossing encoding of `d_t`: UP/DOWN spikes whenever `d_t` moved by θ since the last event (thermometer code, 2 levels per polarity) |
| hidden layer | GRU, H = 64 | recurrent LIF, H = 64, no bias (no input and no spikes = no work) |
| readout | hidden state at the end of 3 time windows + RR | spike counts per neuron in 3 time windows + RR |
| parameters | 13 452 | **4 940** |
| work per beat | 126 steps × 3 gates × (1 + 64) × 64 = 1.57 M MACs | **8.7 input events + 15.5 hidden spikes**, × 64 synapses |

Same protocol as before: every choice by patient-level 5-fold CV on DS1,
3 seeds, DS2 once at the end, macro-F1 primary. Shared choices were tuned
on the GRU (batch 256, lr 3e-3, wd 0.2, augmentation, **20 epochs**: GRU
0.642 vs 0.635 at 10), the readout windows on both (+0.03 for each, so both
use 3), and the RSNN got its own sweep over θ, β and the encoding levels.
All variants: `results/ecg_streaming/pc/cv/ALL.md`.

| CV (DS1, 3 seeds) | macro-F1 |
|---|---|
| GRU H = 16 / 32 / 64, 3 windows, 10 epochs | 0.594 / 0.631 / 0.635 |
| **GRU H = 64, 3 windows, 20 epochs (chosen)** | **0.642 ± 0.015** |
| RSNN θ = 0.2, 1 window, 10 / 20 epochs | 0.594 / 0.609 |
| RSNN θ = 0.2, 3 windows, 10 / 20 epochs | 0.627 / 0.633 |
| RSNN θ = 0.1 / 0.3 / 0.4 / 0.5 / 0.7 / 1.0, 3 windows, 10 epochs | 0.596² / 0.658 / 0.662 / 0.679 / **0.686** / 0.665 |
| RSNN β = 0.95 (θ = 0.2) | 0.629² |
| **RSNN θ = 0.7, 3 windows, 20 epochs (chosen)** | **0.684 ± 0.002** |

² 2 seeds (screening, stopped when clearly not better). The largest single
gain was a *coarser* encoder: θ from 0.2 to 0.7 cut the input events per
beat from ~33 to ~9 and raised the CV macro-F1 by 0.06, i.e. the encoder
also acts as a denoiser for the patient-relative signal.

### 3.3 Accuracy on the held-out patients (DS2, all 49 273 beats)

| (mean ± std over 3 seeds) | GRU | RSNN |
|---|---|---|
| macro-F1, PC float | 0.670 ± 0.005 | 0.668 ± 0.005 |
| macro-F1, int8 (= board, bit-exact) | 0.667 | 0.667 |
| accuracy | 93.05 % | 93.64 % |
| Se / +P S | 19.4 / 33.8 % | 18.1 / 34.0 % |
| Se / +P V | 88.6 / 73.2 % | 89.5 / 72.1 % |
| float → int8 macro-F1 | ≤ 0.006 | ≤ 0.002 |

**On DS2 the two are equally accurate.** The RSNN's clear CV lead
(0.684 vs 0.642) does not carry over to the held-out patients; this is
reported as a draw, not a win. For reference, the feedforward ANN reached
0.675 on DS2.

### 3.4 On the board (bit-exact, every beat)

The GRU costs ~1.1 s per beat on this core, so it runs on a fixed random
subset of 1 000 DS2 beats (all 22 patients); the RSNN runs on the same
beats and, with the coprocessor, also on all 49 273. Both models' accuracy
above comes from all beats (int8 on the PC = board, verified on every beat
the board ran).

| latency per beat | GRU | RSNN | RSNN / GRU |
|---|---|---|---|
| baseline firmware (200 beats, latency only) | 2 203 ms | 35.2 ms | 1 / 63 |
| fast firmware (1 000 beats) | 1 099 ms | **36.4 ms** | **1 / 30** |
| coprocessor build (1 000 beats; GRU: 50) | 1 100 ms | **4.58 ms** | **1 / 240** |
| RSNN, coprocessor, all 49 273 beats × 3 seeds | – | 4.59 ms (median 4.25, p95 6.4, max 14.3) | |
| beats on which the RSNN is faster | | 1 000 / 1 000 per seed | |

- The **GRU is not real-time** on this core at H = 64: 1.1 s per beat is
  longer than a heartbeat at > 55 bpm. Even the GRU with H = 16 (CV 0.594)
  would need ≈ 0.1 M MACs ≈ 70 ms (estimated from 70 cycles per MAC, not
  measured).
- The **RSNN in plain software takes 36 ms**: 126 steps × 0.29 ms, almost
  all of it the LIF update of 64 neurons per step; the synaptic work
  (1 530 SOPs per beat) is small. Fast and baseline firmware are about
  equally fast for it, because their optimisations target synapses.
- With the **coprocessor** a step costs 0.036 ms and the RSNN takes
  **4.6 ms per beat**, slightly faster than the feedforward ANN on the whole
  beat (4.87 ms) at similar accuracy (0.667 vs 0.675 macro-F1), with 42 %
  fewer parameters (4 940 vs 8 515).
- Energy, inferred from cycles (constant power, no measurement): the RSNN
  uses ~30× (software) to ~240× (coprocessor) less energy per beat than the
  GRU.

---

## 4. Interpretation for the research question

> *For edge inference on a resource-constrained processor, is a spiking
> network a better design choice than a classical ANN, and what hardware
> support is worth adding to make it win?*

The three ECG measurements answer the two halves separately:

| setting | SNN vs its ANN counterpart (latency, same core) | accuracy |
|---|---|---|
| feedforward, software only | 1.70× slower | draw |
| feedforward + SNN coprocessor | 1.09× slower at T = 10, faster for T ≤ 5 | draw |
| **streaming, software only** | **30× faster than the GRU** | draw (DS2) |
| **streaming + SNN coprocessor** | **240× faster than the GRU, faster than the feedforward ANN** | draw (DS2) |

1. **The setting matters more than the hardware.** When the input is a
   dense vector (an image, a segmented beat), an SNN on a CPU pays for its
   timesteps and cannot beat a sparse ANN. When the input is a stream with
   long stretches of "nothing new", an event-driven SNN does almost no work
   per sample, while a GRU must evaluate all its gates every sample. That
   alone is a 30× difference on an unmodified RISC-V core.
2. **The hardware worth adding is a neuron-update unit**, not more MACs.
   After event-driven input, what remains is the per-step update of the
   neuron state (leak, integrate, threshold, reset). A small coprocessor
   (+889 LUTs, +2 DSPs, +13 mW estimated, timing still met at 100 MHz)
   makes it ~8× cheaper in the streaming case and lets even the
   feedforward SNN overtake the ANN at T ≤ 5.
3. **Accuracy is a draw throughout**: SNNs did not lose accuracy here, but
   they did not gain any on held-out patients either.

---

## 5. Caveats

- **The GRU's subset.** The GRU ran on 1 000 random DS2 beats (fast) and
  200 (baseline, the first 200 of the sorted sample: almost only N beats,
  so their macro-F1 is meaningless; latency only). Its latency hardly
  varies between beats (dense computation, 1 091 – 1 106 ms), so the subset
  is enough for latency; accuracy comes from all beats.
- **ANN-side event-driven tricks were not tried.** A delta-GRU (skipping
  updates for inputs / hidden units that changed less than a threshold) is
  the ANN analogue of the SNN's event-driven input and would narrow the gap;
  it needs retraining and was out of scope. The GRU got the same firmware
  optimisations as the ANN (sparse inputs, SWAR, blocking), which barely
  help because its hidden state is dense.
- **Same hidden size, not same accuracy budget.** H = 64 for both (the
  task's rule); the RSNN has 2.7× fewer parameters. GRU H = 32 is as
  accurate in CV (0.631 vs 0.642) and would be ~4× cheaper (≈ 280 ms, estimated), still ~60× slower
  than the RSNN on the coprocessor (estimate).
- **Host-side preprocessing** (filtering, R-peak positions, template,
  z-score, level-crossing encoder, int8 quantisation) is identical for both
  and not timed. The encoder is one compare per sample; the template (a
  32-beat median) would cost more than either network on the core.
- **Energy** is inferred from cycles; the Vivado power figures are
  estimates for the whole chip (dominated by the ARM side).
- **Firmware v3.3** replaces v3.1 for the earlier ECG / MNIST models:
  outputs identical to the published runs, cycles within +1.1 % (MNIST,
  vs v3.0) and ±0.04 % (ECG, vs v3.1) on 1 000 images / 2 000 beats
  (`results/snn_coprocessor/v33_regression/`).
- The CV runs of the streaming study were interrupted once (a session
  restart); the affected variants were rerun from scratch, their partial
  logs are kept in `results/ecg_streaming/pc/cv/interrupted_logs/`.

---

## 6. Reproduce

```bash
# Vivado 2026.1 (from code/pynqz2_riscv_flow/vivado/, a path without spaces)
vivado -mode batch -source build.tcl          # stock design   -> spike_top.bit
vivado -mode batch -source build_snn.tcl      # + coprocessor -> spike_top_snn.bit
# coprocessor testbench (code/pynqz2_riscv_flow/sim/)
python gen_vectors.py && xvlog ../rtl/snn_pcpi.v tb_snn_pcpi.v && xelab tb -s tb && xsim tb -R

# firmware (all three levels), from code/ann_vs_snn_mnist/
pynq/build_fw.sh
python pynq/benchmark_pynq.py --fw hw --models SNN         # MNIST SNN on the coprocessor

# ECG, from code/ann_vs_snn_ecg/
python pynq/benchmark_pynq_ecg.py --fw hw --models SNN     # feedforward SNN, coprocessor
S="--use-rr --morph diff --augment --weight-decay 0.2 --batch 256 --lr 0.003 --readout-bins 3 --hidden 64 --epochs 20"
python train_ecg.py --models GRU $S --final --manifest runs/manifest_stream_gru.json
python train_ecg.py --models RSNN $S --delta-theta 0.7 --delta-levels 2 --beta 0.9 \
       --final --manifest runs/manifest_stream_rsnn.json
# (merge both into runs/manifest_stream.json), then
python benchmark_ecg.py --manifest runs/manifest_stream.json --prefix stream_
python pynq/export_stream.py --sample 1000
python pynq/benchmark_pynq_ecg.py --stream --fw fast
python pynq/benchmark_pynq_ecg.py --stream --fw hw --models RSNN
```

## 7. Raw data

| folder | content |
|---|---|
| `results/snn_coprocessor/` | Vivado reports (reference, coprocessor), v3.3 regression runs |
| `results/ann_vs_snn_ecg/` | feedforward ECG incl. `*_hw_*` (coprocessor) runs and `T_study_hw/` |
| `results/ann_vs_snn/` | MNIST incl. `snn_seed*_hw_*` (coprocessor) runs |
| `results/ecg_streaming/pc/` | every CV variant, DS2 PC benchmark (`stream_summary.md`), final manifest |
| `results/ecg_streaming/board_subset/` | GRU and RSNN per-beat board results on the 1 000-beat subset (fast, hw, baseline) |
| `results/ecg_streaming/board_full_hw/` | RSNN on the coprocessor, all 49 273 DS2 beats × 3 seeds |
| `results/ecg_streaming/summary_pynq_stream_*.md` | board benchmark tables per firmware level |
