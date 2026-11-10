# muRISCV-NN as an external ANN reference on the PicoRV32

*Group 4: ANN vs SNN on MNIST with RISC-V. Measured 2026-10-07.*

Our ANN kernels (baseline and fast, [ann_vs_snn_pynq_comparison.md](ann_vs_snn_pynq_comparison.md))
are our own code, so "the ANN is 4.2× faster than the SNN" could just mean
that we optimised the ANN well. As an outside reference we ran the
same int8 network through **muRISCV-NN**, the RISC-V port of Arm's
CMSIS-NN, used **unmodified**, on the same core, bitstream, compiler and
test images.

---

## 1. Headline

| (seeds 0, 1; all 10 000 test images) | ANN, our baseline | **ANN, muRISCV-NN** | ANN, our fast | SNN, our fast |
|---|---|---|---|---|
| board latency [ms/image] | 135.8 (1000 images) | **76.6** (constant) | 13.6 | 57.1 |
| board accuracy | = fast (identical outputs) | **96.85 ± 0.10 %** | 96.84 ± 0.16 % | 97.65 ± 0.11 % |
| bit-exact vs its PC reference | 2 000 / 2 000 | **20 000 / 20 000** | 20 000 / 20 000 | 20 000 / 20 000 |
| cycles per dense MAC | 133.6 | **75.4** | 13.4 | — |

- **muRISCV-NN is 1.77× faster than our baseline ANN and 5.6× slower than
  our fast ANN.**
- **Our fast SNN (57 ms) beats the library ANN (76.6 ms) on 9 998 of
  10 000 images, for both seeds.** The SNN coprocessor level (31.9 ms) is
  2.4× faster than the library ANN.
- So the ANN's lead over the SNN comes from **exploiting input sparsity**,
  which standard dense int8 libraries do not do. A plain int8 CMSIS-style
  ANN would lose to the SNN on this platform.
- The libraries' int8 arithmetic costs no accuracy: muRISCV-NN and our int8
  ANN do not differ significantly (exact McNemar p = 0.58 and 0.41, §4).
- **ECG is different (§7).** The library takes 7.05 ms per beat, so it is
  faster than our fast SNN (8.30 ms) but slower than our fast ANN (4.87 ms)
  and the SNN coprocessor (5.32 ms). ECG inputs are dense (104 of 129
  non-zero), so skipping zeros gains little, and the SNN's T = 10 timesteps
  then cost more than a dense layer.

---

## 2. Setup: what is held equal

| | how |
|---|---|
| library | `third_party/muriscv-nn` (git submodule, tum-ei-eda/muriscv-nn), commit `f69494442858980be78180868d96c3524dbf26f9` (2026-09-07). Its CMSIS-NN base is `4ab83cc3cc98fb85ed6dafb55e8ca02f1628dcae` (commit 4a9a596d "manually sync with cmsisnn ref", 2026-07-28); later commits only touch scripts and tests. Unmodified, Apache-2.0 license kept in the submodule. |
| code path | scalar C only: `USE_VEXT`, `USE_PEXT`, `USE_COREV`, `USE_MNN` and the requantisation inline-asm option undefined (PicoRV32 has no V/P extensions). Every instruction of the linked image is RV32IM (checked on the disassembly). |
| files compiled | `muriscv_nn_fully_connected_s8.c`, `muriscv_nn_fully_connected_get_buffer_sizes_s8.c`, `muriscv_nn_vec_mat_mult_t_s8.c` (plus picolibc headers for its `#include`s) |
| compiler, flags | the firmware's own: gcc-riscv64-unknown-elf 10.2.0 (Docker, Ubuntu 22.04), `-march=rv32im -mabi=ilp32 -O2 …`; the library gets no `-Werror` (one warning: unused `kernel_sum`) |
| network | the same checkpoints and the same int8 weights and int32 biases as our ANN; no retraining |
| hardware | PicoRV32 RV32IM @ 100 MHz (`ENABLE_FAST_MUL`), unchanged `spike_top.bit`, 8-wait-state bus |
| measurement | same timer and timed region as `ann_run`: input copy + all layers + argmax |
| firmware | `snn_mnist_muriscv.bin` = the fast image + "ANN3" models; all changes in `main.c` are under `#if FW_MURISCV`. The baseline, fast and hw images stay byte-identical. |

### 2.1 Mapping our int8 ANN onto CMSIS-NN s8 (no retraining)

| | mapping |
|---|---|
| weights | our symmetric int8 weights, transposed to row-major [n_out][n_in], `filter_offset` 0 |
| input | pixel p as int8 (p − 128), `input_offset` +128, so the kernel sees exactly p, like ours |
| hidden layer | our uint8 ReLU output h as int8 (h − 128): `output_offset` −128, clamp [−128, 127] = ReLU + saturation at 255 |
| biases | int32, identical to ours |
| hidden scale | our real scale m_q / 2²⁴ as a CMSIS Q31 multiplier and shift (TFLite `QuantizeMultiplier`) |
| output layer | **the library's fully-connected kernel only produces int8**, so the logits are requantised to int8: 127 / max \|logit\| over the same 10 000 training images used for the hidden scale. Prediction = first argmax |

Two sources of difference from our int8 ANN, both expected:

- Rounding: CMSIS rounds twice (doubling high multiply, then a rounding
  shift), ours once. Over 10 000 × 128 hidden activations, 213 (seed 0) and
  383 (seed 1) differ by 1 LSB. They change 0 and 1 predictions.
- The int8 logits (ties, saturation): with them, 17 and 26 predictions
  differ from ours. So the int8-only output API causes most of the
  differences.

### 2.2 Verification chain

1. `pynq/muriscv_ref.py`: a numpy model of the library's portable C
   arithmetic, which exports the ANN3 model and the reference logits.
2. `pynq/muriscv/host_ref`: the **same** library sources and the same glue
   (`ann_muriscv.c`), built for the PC, reproduce the numpy logits on
   10 000 / 10 000 images, for both seeds.
3. The board reproduces them **bit for bit**: 10 000 / 10 000 exact
   outputs for each seed.

---

## 3. Latency (same images)

| seed | ANN baseline | ANN muRISCV-NN | ANN fast | SNN fast | muRISCV-NN vs baseline | fast vs muRISCV-NN | images where SNN fast < muRISCV-NN |
|---|---|---|---|---|---|---|---|
| 0 | 135.79 (n = 1000) | 76.62 (min 76.61, max 76.62) | 13.62 | 57.06 | 1.77× faster | 5.62× faster | 9998 / 10000 |
| 1 | 135.79 (n = 1000) | 76.62 (min 76.61, max 76.62) | 13.62 | 57.20 | 1.77× faster | 5.62× faster | 9998 / 10000 |

The library's latency is the same for every image: a dense kernel does not
look at the data.

---

## 4. Accuracy: muRISCV-NN vs our int8 ANN (same images)

| seed | muRISCV-NN | ours | both right | only ours right | only muRISCV-NN right | both wrong | same prediction | McNemar p (exact) |
|---|---|---|---|---|---|---|---|---|
| 0 | 96.92 % | 96.95 % | 9687 | 8 | 5 | 300 | 9983 | 0.581 |
| 1 | 96.78 % | 96.73 % | 9664 | 9 | 14 | 313 | 9974 | 0.405 |

Board test loss (cross-entropy on the rescaled int8 logits):
0.1090 ± 0.0029, the same as our ANN (0.1090 ± 0.0031).

---

## 5. Why: memory accesses decide the speed

The SoC bus has 8 wait states, so every memory access costs about 12
cycles: each instruction fetch, and each data load and store. The
instruction type hardly matters (`mul` is fast with `ENABLE_FAST_MUL`).
From the inner loops in the disassembly (`pynq/muriscv/loops.py`) and the
measured cycles:

| kernel | inner loop | instructions / MAC | data accesses / MAC | memory accesses / MAC | measured cycles / MAC | cycles / access |
|---|---|---|---|---|---|---|
| ours baseline | 1 output: `lb w; lw acc; mul; add; sw acc` + loop | 8 | 3 (accumulator in memory) | 11 | 133.6 (dense) | 12.1 |
| muRISCV-NN | 5 outputs per input: 6 `lb`, 5 `mul`, 6 `add`, 6 `addi`, `bne` | 4.8 | 1.2 (byte loads, accumulators in registers) | 6.0 | 75.4 (dense) | 12.4 |
| ours fast | 8 outputs per non-zero input: 4 `lw` (SWAR, 4 weights each), 4 `mul` + masks | 4.25 | 0.5 | 4.75 | 59.2 per executed MAC (cost-model slope 7 571 / 128) | 12.5 |

The muRISCV-NN row is from 25 groups of 5 rows × 784 inputs × 30 accesses,
plus 3 remaining rows × 784 × 10 accesses, plus layer 2 (2 × 128 × 30):
619 200 accesses for 7.66 M cycles, which is 12.4 cycles per access.

- **muRISCV-NN vs our baseline:** keeping 5 accumulators in registers
  removes the load and store of the accumulator: 11 → 6 accesses per MAC,
  measured 1.77×.
- **Our fast vs muRISCV-NN:** about 1.3× from SWAR (one 32-bit load for 4
  weights instead of 4 byte loads), and about 4.4× from skipping zeros.
  Only 151 of 784 pixels and 82 of 128 hidden units are non-zero, so it
  executes about 20 000 of the 101 632 MACs.
- On a core with cached, single-cycle memory the instruction mix would
  matter more, so the 1.77× and 5.6× are specific to this memory-bound SoC.

---

## 6. Code size

| | text [B] |
|---|---|
| `muriscv_nn_vec_mat_mult_t_s8.o` | 1 768 |
| `muriscv_nn_fully_connected_s8.o` | 316 |
| `muriscv_nn_fully_connected_get_buffer_sizes_s8.o` | 48 |
| glue `ann_muriscv.o` | 724 |
| **library + glue** | **2 856** (0 data) |
| firmware image `snn_mnist_muriscv.bin` | 12 984 (fast image: 9 580, +3 404) |

At `-O3` the glue is 968 B; the library objects keep the same size.
The image ends at 0xCEC0 (including bss), inside the 64 KB program RAM.
The ANN3 `model.bin` is 102 288 B (ours: 103 024 B, padded rows).

---

## 7. ECG: the same reference on MIT-BIH (129-64-3, 3 seeds, all 49 273 DS2 beats)

Same mapping, with two differences. The ECG inputs are already int8
(symmetric, scale s_x from `export_ecg.py`), so `input_offset` is 0. The
int8 output scale is calibrated on the DS1 (training) beats.
`pynq/export_ecg_muriscv.py` first checks that it reproduces our int8 ECG
ANN exactly (the existing `ref_logits.bin`), then maps it.
Verification chain as for MNIST: `host_ref` equals the numpy model on
3 × 49 273 beats, and **the board equals `host_ref` bit for bit on
3 × 49 273 / 49 273 beats**.

| (mean ± std over seeds 0, 1, 2) | ANN, our baseline | **ANN, muRISCV-NN** | ANN, our fast | SNN, our fast | SNN coprocessor |
|---|---|---|---|---|---|
| board latency [ms/beat] | 11.70 | **7.05** (constant) | 4.87 | 8.30 | 5.32 |
| board macro-F1 (N/S/V) | 0.675 ± 0.004 | **0.677 ± 0.004** | 0.675 ± 0.004 | 0.706 ± 0.007 | 0.706 |
| board accuracy | 93.62 % | **93.69 ± 0.10 %** | 93.62 ± 0.12 % | 94.18 % | 94.18 % |
| cycles per dense MAC | 138.5 | **83.5** | 57.7 | — | — |

Per seed, on the same beats (`compare_pynq.py --task ecg`, §6–8): muRISCV-NN
is 1.66× faster than our baseline and our fast ANN is 1.44–1.45× faster than
muRISCV-NN. Our fast SNN is faster than the library on only 3 231 / 3 968 /
3 729 of 49 273 beats.

- **Why the ranking flips compared with MNIST.** The fast ANN's lead over
  the library shrinks from 5.6× to 1.45×, because the ECG inputs are dense:
  104 of 129 features and 38.5 of 64 hidden units are non-zero. What remains
  is mostly the SWAR loads. The library has a higher cost per MAC than on
  MNIST (83.5 vs 75.4 cycles). On a net this small, the fixed part (per-row
  requantisation, the 3-output layer in the 1-row remainder loop, calls)
  weighs more: about 53 500 memory accesses in the inner loops for
  705 000 cycles, i.e. 13.2 cycles per access.
- **Accuracy: a significant difference that is an artefact.** On ECG,
  muRISCV-NN gets more beats right than our int8 ANN (69 / 44 / 37 vs
  20 / 16 / 7 discordant beats, exact McNemar p < 0.001 on every seed).
  But 225 of the 226 differing beats have **tied int8 logits**. The first
  argmax then picks the lowest class index, mostly N, which is the majority
  class (90 % of DS2). Without the ties the two networks predict the same
  beats (1 beat differs, on seed 1). The macro-F1 rises by only +0.002. So
  this is a side effect of the int8-only output API plus first-index
  tie-breaking, not better arithmetic.
- **Regression.** Our ECG ANN fast and SNN fast (seed 0, 200 beats, current
  v3.3 images) reproduce the committed predictions, outputs and events on
  200 / 200 beats. Cycles are within +0.006 % (ANN) and −0.042 % (SNN) of
  the committed run.

---

## 8. Caveats

- **Firmware version.** The committed baseline and fast results in
  `docs/results/ann_vs_snn/` were measured with firmware v3.0 (2026-10-01/02).
  The muRISCV-NN image is built on v3.3, which has the ECG additions. The
  current v3.3 images are 0.17 % (ANN baseline), 1.06 % (ANN fast) and
  0.08 % (SNN fast) slower than v3.0, with identical outputs. Rerunning the
  v3.0 image today reproduces the committed cycles exactly (200 / 200). So
  the difference comes from the code layout, not from the board, and it is
  far below the gaps in §3.
- **Scalar path only.** muRISCV-NN's speed comes from the RISC-V vector (V)
  and packed SIMD (P) extensions, which PicoRV32 does not have. This is a
  reference for "a standard int8 library in portable C", not for
  muRISCV-NN on a vector core.
- **int8 logits.** The library has no int32-output fully-connected layer,
  so its argmax works on int8 logits (37 / 48 tied top logits). This does
  not change the accuracy significantly (§4).
- **No sparsity in the reference.** This is what the library offers, not a
  flaw in the mapping. A sparse variant would mean changing its kernels,
  which a valid reference must not do.

---

## 9. Reproduce

From `code/ann_vs_snn_mnist/` in Git Bash, venv active, board reachable:

```bash
git submodule update --init third_party/muriscv-nn   # once (from the repo root)
python pynq/muriscv_ref.py                           # map + export both seeds
pynq/muriscv/build_muriscv.sh                        # host_ref + RV32IM size / opcode check
pynq/build_fw.sh                                     # all images incl. snn_mnist_muriscv.bin
python pynq/benchmark_pynq.py --fw muriscv           # board, both seeds (~30 min)
python pynq/compare_pynq.py --results ../../docs/results/ann_vs_snn   # tables 6-8
```

Check the host build of the library against the numpy model (the
`*_muriscv` export folders):

```bash
pynq/muriscv/check_host_ref.sh pynq/export/ann_mnist_seed{0,1}_muriscv
```

ECG, from `code/ann_vs_snn_ecg/` (venv active, after `export_ecg.py`):

```bash
python pynq/export_ecg_muriscv.py                    # map + export 3 seeds
../ann_vs_snn_mnist/pynq/muriscv/check_host_ref.sh pynq/export/ann_ecg_seed{0,1,2}_muriscv
python pynq/benchmark_pynq_ecg.py --fw muriscv       # board, 3 seeds (~25 min)
python ../ann_vs_snn_mnist/pynq/compare_pynq.py --task ecg --results ../../docs/results/ann_vs_snn_ecg
```

## 10. Raw data

MNIST, in `docs/results/ann_vs_snn/`:

| file | content |
|---|---|
| `ann_seed{0,1}_muriscv_board_results.csv` | one row per test image (10 000), same columns as the other levels |
| `ann_seed{0,1}_muriscv_board_summary.json` | per-run summary from the board |
| `ann_seed{0,1}_muriscv_meta.json` | mapping (multipliers, shifts, offsets), PC accuracies, agreement with our int8 ANN |
| `summary_pynq_muriscv.{md,json}` | benchmark table (mean ± std over seeds) |
| `comparison_tables.md` §6–8 | output of `compare_pynq.py` |

ECG, in `docs/results/ann_vs_snn_ecg/`:

| file | content |
|---|---|
| `ann_seed{0,1,2}_muriscv_board_results.csv.gz` | one row per DS2 beat (49 273), gzipped like the other ECG results |
| `ann_seed{0,1,2}_muriscv_board_summary.json` | per-run summary from the board |
| `ann_seed{0,1,2}_muriscv_meta.json` | mapping, PC metrics (float, ours int8, muRISCV-NN), agreement, ties |
| `summary_pynq_muriscv.{md,json}` | benchmark table with confusion matrix and per-record results |
| `comparison_tables.md` §6–8 | output of `compare_pynq.py --task ecg` |
