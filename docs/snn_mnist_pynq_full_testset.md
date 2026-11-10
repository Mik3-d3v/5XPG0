# MNIST SNN on the PYNQ-Z2: full test set and the optimised firmware

*Group 4: ANN vs SNN on MNIST with RISC-V. Run of 2026-10-01.*

This document follows [snn_mnist_pynq_deployment.md](snn_mnist_pynq_deployment.md),
which covers the setup, quantization, input path and the first (baseline)
run. Here: why the baseline was too slow for the full test set, what was
changed to make the full benchmark take **~10 minutes per seed instead of
~67**, and the board results on all 10 000 MNIST test images for both seeds.

> **Tooling changed after this document (ANN support, see
> [ann_vs_snn_pynq_comparison.md](ann_vs_snn_pynq_comparison.md)).**
> `export_snn.py` is now `export_model.py` (exports ANN and SNN). Both
> firmware images (v3.0) run either model, chosen by the `model.bin`
> magic, and the mailbox command is `MB_CMD_RUN`. The board CSVs now use
> generic columns (`outputs_match`, `in_events`, `in_changes`,
> `hidden_events`, `out_events`, `outputs`), and `benchmark_pynq.py`
> writes `runs/summary_pynq_<fw>.md`. The SNN results are unchanged: the
> v3.0 firmware reproduces them bit for bit, within a few cycles.

---

## 1. Summary

| | value |
|---|---|
| models | SNN 784-128-10 LIF, T = 25, seeds 0 and 1 (`runs/manifest.json`) |
| test images | **all 10 000**, per seed, same spike trains as `benchmark.py` |
| board test accuracy | **97.65 ± 0.11 %** (seed 0: 97.72 %, seed 1: 97.57 %) |
| PC float accuracy (`benchmark.py`) | 97.64 ± 0.07 % |
| correctness | **20 000 / 20 000** samples: output spike counts identical to the PC int8 reference |
| latency (fast firmware) | **57.13 ± 0.10 ms/sample** (5.71 M cycles at 100 MHz) |
| latency (baseline firmware) | ~400 ms/sample |
| speed-up | **7.1×** (same 3077 samples), bit-identical results |
| wall time, full test set | **9.8 min per seed** (was ~67 min) |
| platform | unchanged bitstream, plain RV32IM, no custom instructions |

---

## 2. Starting point: the baseline was too slow, and loading was not the cause

The baseline firmware needs ~0.4 s per sample, so 10 000 images take ~67 min
per seed. A full run of the baseline was started and **stopped after 3077
samples** to make room for this work. Its data is kept in
`docs/results/full_run_partial/`:

| baseline v1.0, seed 0 | value |
|---|---|
| samples | 3077 (test images 0 to 3076) |
| board accuracy | 97.11 % (PC float on the same images: 96.98 %) |
| board == PC int ref | 3077 / 3077 exact counts |
| latency | 401.8 ms/sample |
| host overhead | 0.69 ms/sample |

**Data loading was not the bottleneck.** The host overhead per sample
(reading the sample from `inputs.bin`, copying 2500 B over AXI into BRAM,
the mailbox round trip and reading the result) is **0.67–0.72 ms, about 0.2 %
of the baseline latency**. Making it faster could not change the total run
time. The RISC-V compute was the bottleneck.

The loading path was still made more robust for long runs:

- inputs are streamed from file instead of read whole;
- results are appended per sample and can be resumed (`--resume`);
- a detached mode runs on the board with `nohup` (`--detach`, `status_pynq.sh`);
- unchanged files are skipped on upload (md5).

### Constraints for the optimisation

- **No bitstream change was possible.** The only Vivado here is 2026.1,
  which dropped Zynq-7000, so the fixed 8-wait-state bus stays. This also
  keeps the platform identical for the other group.
- **No custom instructions** (PCPI), as requested.
- The results must stay **bit-exact**, so that board and PC can still be
  compared sample by sample.

---

## 3. The optimisations (firmware `snn_mnist_fast.bin`)

Both images are built from the same `firmware/main.c`. The baseline is
`SNN_FAST=0` and the fast kernel `SNN_FAST=1`. The model, input format and
results are identical.

### 3.1 Why the baseline costs ~122 cycles per synapse

The baseline adds one int8 weight row per input spike, byte by byte: 7
instructions per synapse (`lb, lw, add, sw` + 3 loop instructions).
PicoRV32 is not pipelined, and every instruction fetch and every load goes
through the BRAM bus with `READ_WAIT = 8`, so each instruction costs
≈ 17 cycles. Cost therefore scales with **instructions and memory accesses
per synapse**, and both had to go down.

### 3.2 Delta currents (exploits temporal sparsity, exact)

The input current is linear in the input spikes, so it can be updated
incrementally:

```
I_t = I_{t-1} + Σ W[inputs that turned on]  −  Σ W[inputs that turned off]      (I_{-1} = b)
```

This is exact in integers, with no approximation. It pays off because of
how rate coding behaves on MNIST. Saturated pixels (value ≈ 1) spike in
**every** step, and black pixels never do. Only grey pixels toggle. Measured
on the test set:

| per sample (25 steps) | count |
|---|---|
| active input spikes | 2 598 |
| input changes (on + off) | 802 (**31 %** of the active inputs) |

So layer 1 only processes about a third of the rows. The same code also
runs for layer 2. There it gains nothing, because hidden spikes rarely
repeat (12.6 changes vs 14.6 active per step), but it costs little.

The changed inputs come from a bitmap diff (`x ^ prev`), and the set bits
become index lists with a de Bruijn count-trailing-zeros (multiply + table)
instead of a bit-by-bit loop.

### 3.3 SWAR accumulation (4 weights per load, 16 outputs per pass)

At boot the int8 weights are converted in place to offset binary
(`u = w + 128`, `0..255`). One 32-bit load then fetches the weights of 4
consecutive outputs, which are split into two registers with 16-bit lanes:

```c
w = p[0];  a0 += w & 0x00FF00FF;          // outputs 0 (lo) and 2 (hi)
           a1 += (w >> 8) & 0x00FF00FF;   // outputs 1 (lo) and 3 (hi)
```

- 8 registers hold 16 outputs, so a pass over the changed rows touches no
  memory except the weights.
- A lane holds at most 255 · n, so it is flushed every ≤ 256 rows
  (255 · 257 < 65 536). The flush subtracts `128 · n` to undo the offset.
  "Off" rows are subtracted in the flush.
- This is ~1.5 instructions per synapse instead of 7, and 1 load per 4
  synapses instead of 2 per synapse.
- Weight rows are padded to a multiple of 16 bytes (model format `"SNN2"`,
  `w_stride`), so the word loads are aligned. PicoRV32 traps on misaligned
  loads.

### 3.4 Small things

- The reset uses `mem_{t-1} > thr`, which equals "spiked last step" for
  thr > 0, so no separate spike array is needed. The baseline was switched
  to this too.
- Output bitmaps feed the next layer directly.

### 3.5 What was *not* done

- **Bus wait states.** `READ_WAIT = 8` in `rtl/spike_soc.v` is now the
  dominant cost (§5.2). BRAM read latency is 1 cycle, so a rebuild with a
  smaller value would probably be the single biggest gain. It needs Vivado
  2024.1 and changes the shared platform.
- Custom instructions (the planned ISA-extension part of the project).
- Fewer timesteps or a different encoding. Those would change the model,
  not the implementation.

---

## 4. Results on the full test set

`python pynq/benchmark_pynq.py` (fast firmware, 10 000 images per seed),
next to `benchmark.py` on the PC:

| metric (mean ± std over seeds 0, 1) | PC (`benchmark.py`) | **PYNQ-Z2 RISC-V** |
|---|---|---|
| test accuracy [%] | 97.64 ± 0.07 | **97.65 ± 0.11** |
| test loss (CE on spike counts) | 0.0776 ± 0.0026 | 0.0772 ± 0.0029 |
| parameters | 101 770 | 101 770 (int8 weights, int32 biases) |
| SOPs/sample | 336 614 ± 407 | 336 608 ± 402 |
| hidden spikes/sample | 412.9 ± 43.1 | 412.3 ± 42.7 |
| output spikes/sample | 22.9 ± 0.7 | 22.9 ± 0.6 |
| latency [ms/sample] | 0.27 (PC CPU, batched) | **57.13 ± 0.10** (PicoRV32, 100 MHz) |
| cycles/sample | n/a | 5 712 887 ± 10 229 |
| model memory | n/a | 103 024 B BRAM |
| board == PC int8 reference | n/a | **100.00 %** (10 000 / 10 000 per seed) |

Per seed:

| | seed 0 | seed 1 |
|---|---|---|
| board accuracy | 97.72 % | 97.57 % |
| PC float accuracy (same spikes) | 97.69 % | 97.59 % |
| board vs float predictions agree | 9 986 / 10 000 | 9 982 / 10 000 |
| latency mean / median | 57.06 / 57.62 ms | 57.20 / 57.71 ms |
| latency min / max | 23.0 / 78.5 ms | 23.8 / 79.7 ms |
| SOPs/sample | 336 324 | 336 892 |
| host overhead | 0.67 ms | 0.67 ms |
| wall time | 589 s (9.8 min) | 590 s (9.8 min) |

Reading the table:

- **Quantization to int8 costs nothing.** The board accuracy is within
  0.03 % of float on each seed. The float/int8 predictions differ on only
  14–18 of 10 000 images, in both directions.
- Spike counts and SOPs on the board match the PC benchmark to within the
  rounding of the integer membrane. The tiny differences (412.3 vs 412.9
  hidden spikes) come from quantization, not from the port. `benchmark.py`
  uses the same input spikes.

---

## 5. Performance analysis

### 5.1 Speed-up

| firmware | ms/sample (seed 0) | samples compared | speed-up |
|---|---|---|---|
| baseline v1.0 | 401.8 | first 3077 | 1× |
| fast v2.0 | 56.6 | same 3077 | **7.1×** |
| baseline v2.0 / fast v2.0 | 375.1 / 55.5 | first 50 | 6.8× (per sample 3.4× to 12.8×) |

The speed-up varies per image. Images with many saturated pixels gain the
most from the delta trick.

### 5.2 Where the remaining cycles go

A least-squares fit over all 10 000 seed-0 samples (R² = 0.995):

```
cycles ≈ 3450 · (input changes) + 124 · (hidden spikes) + 2.89 M
```

- **Per input change: 3450 cycles** for 128 synapses, i.e. **27 cycles per
  synapse** (was 122). This is 49 % of the time.
- **Fixed: 2.89 M cycles = 28.9 ms per sample (51 %)**, i.e. ~116 k cycles
  per timestep. It goes to the LIF update of 138 neurons (64-bit multiply
  for the Q16 leak), the lane flushes (16 lanes × 8 blocks), the bitmap diff
  of 25 words, and per-sample initialisation. All of it is limited by
  instruction fetch over the 8-wait-state bus.

Further software work (one combined on/off pass, 32 outputs per pass, a
32-bit fast path for the leak) would gain maybe 20–30 %. A smaller
`READ_WAIT` would speed up *everything*, which is why it is the next lever.

### 5.3 Notes for the report (ANN vs SNN)

- **The SOP metric and the work actually done now differ.** The SOP column
  (336 k, as defined in `benchmark.py`) counts one add per spike per fan-out
  synapse. The fast firmware performs about 802 × 128 + 412 × 10 ≈ 107 k
  synaptic adds per sample: only the *changes* of the input layer, plus the
  hidden spikes. That is close to the 101 632 MACs of the dense ANN. When
  comparing ANN and SNN, state which cost model is used.
- **Temporal sparsity is a real SNN advantage on this hardware.** It comes
  from rate coding plus mostly saturated MNIST pixels. It depends on the
  encoding: latency or delta encoding would change it.
- **A fair ANN comparison needs an equally optimised ANN.** SWAR
  (4×int8 per load) applies to the ANN's dense MACs too, but needs
  multiplies or a different trick, since the ANN inputs are not binary. Both
  should be measured with the same level of effort, on the same unchanged
  bitstream.
- Board latency is deterministic (same image → same cycles), so the error
  bars in the table come only from the seed.

---

## 6. Reproduce

From `code/ann_vs_snn_mnist/` in Git Bash, with the venv active:

```bash
pynq/build_fw.sh                        # both firmwares (Docker if no riscv gcc)
python pynq/benchmark_pynq.py           # export if needed, run all seeds, summarize
#   --n 1000          fewer images       --fw baseline   the slow kernel
#   --resume          continue a run     --no-run        only re-summarize
```

Output: `runs/summary_pynq.md`, `runs/summary_pynq.json` and per seed
`pynq/export/<name>/board_results_fast.csv` / `board_summary_fast.json`.
The board needs nothing besides the stock PYNQ 3.1 image (see the
deployment doc).

---

## 7. Raw data (`docs/results/`)

| file | content |
|---|---|
| `full_testset/seed{0,1}_fast_board_results.csv` | one row per test image: label, board / PC-int / PC-float prediction, `counts_match`, cycles, input / hidden / output spikes, host ms, output counts |
| `full_testset/seed{0,1}_fast_board_summary.json` | per-seed summary (as printed on the board) |
| `full_testset/seed{0,1}_meta.json` | quantization scales, thresholds, PC accuracies |
| `full_testset/summary_pynq.{md,json}` | mean ± std table from `benchmark_pynq.py` |
| `full_testset/summary_pc_benchmark.md` | the PC `benchmark.py` table used for comparison |
| `full_testset/seed0_baseline_first50_board_results.csv` | baseline v2.0 on the first 50 images (speed-up check) |
| `full_run_partial/board_results.csv`, `run.log` | the stopped baseline v1.0 full run (3077 images) |
| `first_run/` | the first 1000 / 100-image runs (see deployment doc) |
