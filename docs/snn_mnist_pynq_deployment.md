# MNIST SNN on the PYNQ-Z2 RISC-V core: setup, design decisions, first results

*Group 4: ANN vs SNN on MNIST with RISC-V. First on-board run, 2026-10-01.*

This document records how the trained snnTorch SNN from
`code/ann_vs_snn_mnist/` was brought onto the PicoRV32 soft core of the
PYNQ-Z2: what was built, why it was built that way, and what came out. It is
meant as source material for the report and as a log for later work. The run
on the full test set, and the optimised firmware that makes it take minutes,
are documented in [snn_mnist_pynq_full_testset.md](snn_mnist_pynq_full_testset.md).

> **Changes since this first run (same day).** The model format is now
> `"SNN2"`: each layer descriptor has a 7th word, `w_stride`, and weight rows
> are zero-padded to a multiple of 16 bytes (model 103 024 B instead of
> 102 248 B). The baseline firmware (`snn_mnist.bin`, v2.0) now derives the
> reset from `mem_{t-1} > thr` instead of keeping a spike array. This is
> mathematically identical (thr > 0, mem starts at 0) and still bit-exact.
> A second, fast firmware (`snn_mnist_fast.bin`) exists. The numbers in
> this document are from the v1.0 baseline.

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

## 1. Scope

**Goal:** a first, basic, *correct* run of the trained MNIST SNN weights on
the RISC-V core, fed with the same encoded input data as the software
benchmark (`benchmark.py`).

**Deliberately out of scope for this baseline:**

- custom instructions or PCPI coprocessors;
- RTL or bitstream changes;
- any speed optimisation of the inference code;
- the ANN path.

This keeps the result a clean **plain RV32IM baseline**. Later work (ISA
extensions, optimisations, the ANN) can be measured against it.

---

## 2. Setup

### 2.1 Hardware

| item | value |
|---|---|
| board | TUL PYNQ-Z2 (Zynq-7020, xc7z020clg400-1), 491 MB RAM visible to Linux |
| board image | PynqLinux (Ubuntu 22.04), PYNQ 3.1.1, Python 3.10.4, kernel 6.6.10-xilinx-v2024.1 |
| board IP | 192.168.2.99 (`code/pynqz2_riscv_flow/.board-ip`) |
| bitstream | `code/pynqz2_riscv_flow/vivado/spike_top.bit`, **unchanged** prebuilt |
| soft core | PicoRV32, RV32IM (`ENABLE_FAST_MUL=1`, `ENABLE_DIV=1`, no IRQ, no PCPI used) |
| core clock | 100 MHz (FCLK0) |
| memory | one 256 KB true dual-port BRAM: port A = PS over AXI-Lite, port B = CPU (code, data, I/O buffers) |
| CPU bus | every BRAM read takes `READ_WAIT = 8` extra cycles (`rtl/spike_soc.v`); there are no caches |

### 2.2 Software (PC side)

| item | version |
|---|---|
| OS | Windows 11; scripts run in Git Bash |
| Python (training venv) | 3.13, torch 2.14.0+cpu, torchvision 0.29.0, snntorch 1.0.0, numpy 2.5.3 |
| RISC-V compiler | `riscv64-unknown-elf-gcc` 10.2.0 (Ubuntu 22.04 package), run in Docker (`ubuntu:22.04`), because no RISC-V toolchain was installed on Windows/WSL |
| compile flags | `-march=rv32im -mabi=ilp32 -O2 -ffreestanding -nostdlib -fno-tree-loop-distribute-patterns` |
| repo state | based on commit `50ef0a5` plus the uncommitted MNIST training changes |

### 2.3 Trained models used

From `runs/manifest.json` (`train_mnist.py`: 5 epochs, seeds 0 and 1,
batch 64, lr 1e-3, Adam + cosine schedule):

| | value |
|---|---|
| topology | 784 → 128 LIF → 10 LIF (`fc1/lif1/fc2/lif2`), 101 770 parameters |
| LIF | snnTorch `Leaky`, β = 0.9, threshold = 1.0, reset by subtraction, `reset_delay=True` |
| encoding | rate (Bernoulli) encoding, T = 25 timesteps |
| readout | class = argmax of the output spike counts over T steps |
| PC test accuracy (float) | seed 0: 97.69 %, seed 1: 97.59 % (matches `runs/summary.md`: 97.64 ± 0.07 %) |

### 2.4 Files

All new code is in `code/ann_vs_snn_mnist/pynq/`. Nothing in
`code/pynqz2_riscv_flow/` was changed, so the demo and smoke test of the other
group still work.

| file | runs on | purpose |
|---|---|---|
| `export_snn.py` | PC | quantize checkpoints, encode the test inputs, bit-exact integer reference, PC accuracies |
| `benchmark_pynq.py` | PC | benchmark.py on the board: all seeds, mean ± std (added later) |
| `firmware/main.c` | RISC-V | integer LIF network, driven by mailbox commands; builds baseline and fast images |
| `firmware/snn_layout.h` | RISC-V | BRAM layout and mailbox command numbers (mirrored in the Python scripts) |
| `firmware/Makefile`, `build_fw.sh` | PC | build `snn_mnist.bin` (reuses the flow's `start.S`, `linker.ld`, `board.h`, `hal.h`) |
| `host/run_mnist_pynq.py` | board (PS) | load bit + firmware + model, stream inputs, collect and compare results |
| `deploy_pynq.sh` | PC | upload changed files over SSH, run (interactive or `--detach`) |
| `status_pynq.sh` | PC | follow a detached run and copy its results back |
| `README.md` | n/a | quick-start |

### 2.5 How to reproduce

From `code/ann_vs_snn_mnist/` in Git Bash:

```bash
source .venv/Scripts/activate
python train_mnist.py                     # (already done) -> runs/*.pt + manifest
python pynq/export_snn.py                 # 1000 test images per SNN (add --n-samples 10000 for all)
pynq/build_fw.sh                          # only needed after editing the firmware
pynq/deploy_pynq.sh 192.168.2.99 snn_mnist_seed0             # interactive run
pynq/deploy_pynq.sh 192.168.2.99 snn_mnist_seed0 --detach    # long run in the background
pynq/status_pynq.sh 192.168.2.99 snn_mnist_seed0             # progress + fetch results
```

The board prerequisites are the same as for `pynqz2_riscv_flow/install.sh`:
SSH key auth for `xilinx@<ip>` and passwordless sudo.

---

## 3. System overview

```
 PC (Windows)                          PYNQ-Z2
 ─────────────                         ─────────────────────────────────────────────────
 runs/snn_mnist_seed<k>.pt             PS (ARM, Linux)               PL (FPGA fabric)
        │                              run_mnist_pynq.py             ┌──────────────────────┐
 export_snn.py ──────────────┐           │  1 load spike_top.bit     │ ps_if (AXI-Lite)     │
   quantize -> model.bin     │  scp      │  2 write snn_mnist.bin ──▶│   │ port A           │
   encode   -> inputs.bin    ├────────▶  │  3 write model.bin ──────▶│ 256 KB BRAM          │
   int ref  -> ref_counts    │           │  4 per sample:            │   │ port B           │
   labels, float preds       │           │    write spikes ─────────▶│ PicoRV32 (RV32IM)    │
 build_fw.sh -> snn_mnist.bin┘           │    mailbox RUN(T) ───────▶│   runs main.c        │
                                         │    ◀── pred, cycles,  ────│   LIF layers         │
                                         │        counts             │ timer (100 MHz)      │
 status_pynq.sh ◀── csv/json ─────────   │  5 compare with labels    └──────────────────────┘
                                         │    and PC int reference
```

---

## 4. Design decisions

### 4.1 Reuse the unchanged bitstream, keep the firmware separate

The bring-up flow already provides everything the first run needs. The PS can
write any BRAM address over AXI, and the CPU sees the same BRAM. A
console ring and a mailbox already exist. A real input path therefore needs
**no RTL change**: it is just another BRAM buffer that the PS writes.
Keeping the RTL unchanged also avoids a ~15 min Vivado rebuild and keeps the
reference design intact.

The firmware lives in its own folder instead of replacing
`pynqz2_riscv_flow/firmware/main.c`, because that flow is shared and its
smoke test checks the demo neuron. The new firmware reuses the flow's
`start.S`, `linker.ld`, `board.h` and `hal.h` through the Makefile, so the
memory map cannot drift apart.

### 4.2 Quantization is necessary, not optional

Two facts force integer weights:

1. **Memory.** In float32, 784·128 + 128·10 weights + biases ≈ 101.8 k
   values ≈ 407 KB. The entire BRAM is 256 KB, and it also has to hold code,
   stack and buffers.
2. **No FPU.** PicoRV32 RV32IM has no floating point unit. Soft-float
   through libgcc would be very slow and adds code.

**Scheme:** symmetric, per-layer, post-training quantization, with no retraining.

| quantity | stored as | formula |
|---|---|---|
| weights | int8, input-major `[n_in][n_out]` | `W_q = clip(round(W / s), -127, 127)`, `s = max|W| / 127` per layer |
| biases | int32 | `b_q = round(b / s)` |
| threshold | int32 | `thr_q = round(thr / s)` (seed 0: 507 / 661, seed 1: 499 / 557 for layer 1 / 2) |
| β (decay) | Q16 int | `beta_q16 = round(β · 65536)` = 58982 |
| membrane | int32 | in units of `s` |

The scale cancels within each layer: the membrane, bias, weights and
threshold all share `s`, so `mem > thr` gives the same decision as in float.
Spikes between layers are binary (0/1), so **the scales of different
layers never interact**. No requantization is needed between layers. This
is a property of SNNs that an int8 ANN would not have, since it has to
rescale activations.

Model size: 102 248 B (`model.bin`), which fits in the 176 KB model region.

**Accuracy cost: none measurable** (full test set, same input spikes):

| | float | int8 | predictions identical |
|---|---|---|---|
| seed 0 | 97.69 % | 97.72 % | 99.86 % |
| seed 1 | 97.59 % | 97.57 % | 99.82 % |

### 4.3 Integer LIF that matches snnTorch exactly

snnTorch 1.0 `Leaky` (defaults, `reset_delay=True`, `reset_mechanism="subtract"`)
computes, per step:

```
reset_t = (mem_{t-1} > thr)              # = the spike of the previous step
mem_t   = β·mem_{t-1} + I_t − reset_t·thr
spk_t   = (mem_t > thr)                  # strict '>'
```

The firmware implements exactly this in integers:

```c
m = (s32)(((s64)beta_q16 * mem) >> 16) + cur;   // leak (64-bit mul: mul + mulh)
if (spk_prev) m -= thr;                         // reset by subtraction
spk = m > thr;
```

Here `I_t = b + Σ W[:, j]` over the active inputs `j`. The bias is added
every timestep, as in `nn.Linear`. The `>> 16` is an arithmetic shift, which
rounds towards −∞. This is the only rounding difference to float and is
modelled identically on the PC. The state starts at zero for each sample,
like `init_leaky()`. Prediction = **first** argmax of the output spike counts,
the same tie-breaking as `torch.argmax`. 9 of 1000 samples had ties.

### 4.4 A bit-exact PC reference as the correctness oracle

`export_snn.py` contains `int_forward()`, a NumPy model of the firmware using
exactly the same integer operations. It runs over the same input spikes and
stores the output spike counts per sample (`ref_counts.bin`). The board must
reproduce these counts **exactly**, not just the predicted class.

This separates two questions that would otherwise be mixed up:

- *Is the port correct?* Compare board vs PC integer counts. Anything
  other than 100 % identical is a bug.
- *What does quantization cost?* Compare PC integer vs PC float, which needs
  no hardware.

### 4.5 Input path: encode on the PC, identical to the software benchmark

Requirement: the board gets the same encoded inputs as the software
benchmark.

- `benchmark.py` encodes the **test set** with `rate_encode` in batches of
  256 (`shuffle=False`) after `torch.manual_seed(seed)`. The exporter repeats
  exactly this loop, so the random stream and therefore every spike are
  identical. Check: the float accuracy computed by the exporter (97.69 / 97.59 %)
  equals `benchmark.py`'s result to two decimals.
- Encoding on the board is not an option. There is no torch on the 32-bit
  ARM image, and a different RNG would produce different spikes, which breaks
  the comparison. The hardware Poisson generator of the flow
  (`rtl/poisson.v`) is **not used**. It has only 8 channels and its own LFSRs.
- **Format:** per sample `[T=25][25]` uint32 words. Bit `j % 32` of word
  `j / 32` is input pixel `j`, padded from 784 to 800 bits. That is
  2500 B per sample and 25 MB for the full test set.
- **Transfer per sample:** the PS writes 2500 B to `INPUT_BASE`, then sends
  mailbox command `MB_CMD_SNN_RUN(T)`. The firmware replies with
  `(pred, cycles, spikes, err)` and leaves the output counts at `RESULT_BASE`.
  One sample is processed per command. This keeps the protocol trivial.

### 4.6 Event-driven accumulation, input-major weights

For each timestep the firmware turns the input bitmap into a list of active
indices. It then adds one weight **row** per active input to the current
vector. The weights are stored input-major (`[n_in][n_out]`) so each row is
contiguous. The work is therefore proportional to the number of spikes, which
is the synaptic-operation (SOP) cost model of `benchmark.py`. A spiking
hidden layer feeds layer 2 the same way.

This is simply the natural way to write an SNN, not an optimisation. A
dense loop over all 784 inputs would waste most of its time on zeros. The
average input density is about 12 %.

### 4.7 Memory map (BRAM, RISC-V address = PS offset from 0x4000_0000)

| range | content | owner |
|---|---|---|
| 0x00000 – 0x0FFFF | firmware: 2.2 KB code + 28 KB bss (state arrays) | linker.ld |
| 0x10000 | console ring (512 B) | flow `board.h` |
| 0x10400 | mailbox (seq_in, seq_out, cmd[4], resp[4]) | flow `board.h` |
| 0x10800 | `out_counts[n_out]` u32 | CPU writes |
| 0x11000 – 0x12FFF | input spikes of one sample (max 8 KB) | PS writes |
| 0x13000 – 0x3EFFF | `model.bin` (max 176 KB) | PS writes before CPU reset release |
| 0x3F000 – 0x3FFFF | stack (4 KB) | CPU |

The flow's spike-log ring (0x11000) is not used, and its space is reused for
the inputs.

### 4.8 Generic model format

`model.bin` begins with a header: magic `"SNN1"`, layer count, `n_in`, T,
then per layer `n_in, n_out, beta_q16, thr_q, w_off, b_off`. The firmware
reads all sizes from this header and validates them at boot. It supports up
to 4 layers with up to 1024 neurons each. **A different hidden width or
depth only needs a re-export, not a firmware rebuild.** The BRAM limit is
about H ≤ 220 for a 784-H-10 network. The exporter finds layers by the
`fc<i>` / `lif<i>` naming of `train_mnist.py`.

### 4.9 Smaller decisions

- `-fno-tree-loop-distribute-patterns`: without it, gcc turns the zeroing
  loops into `memset` calls, which do not exist with `-nostdlib`.
- `out_counts` is `volatile`, so the result stores are ordered before the
  mailbox reply store.
- Docker build: no RISC-V gcc was installed, and Docker Desktop was already
  running. `build_fw.sh` uses a local `riscv64-unknown-elf-gcc` if one is on
  `PATH`.
- Latency is measured on the board with the 100 MHz free-running timer
  around `run_sample()`. It excludes the PS-side transfer, which is measured
  separately as host overhead.

---

## 5. Results of the first run

### 5.1 Correctness and accuracy

| run | samples | board acc. | PC float acc. (same samples) | board = PC int ref (exact counts) |
|---|---|---|---|---|
| seed 0 | first 1000 test images | **98.30 %** | 98.10 % | **1000 / 1000** |
| seed 1 | first 100 test images | 100.00 % | 100.00 % | **100 / 100** |

- The port is **bit-exact**: every output spike count of every sample equals
  the PC integer reference.
- Board vs float disagreed on 2 of 1000 samples (quantization). Here the
  board was right more often, but on the full test set the int8 and float
  accuracies are within ±0.03 %.
- The first 1000 test images are easier than the average: 98.1 % float here
  vs 97.69 % on all 10 000.
- The cycle counts are deterministic. Repeating the first 20 samples gave
  identical cycle counts.

### 5.2 Latency on PicoRV32 (100 MHz)

| | seed 0 (1000) | seed 1 (100) |
|---|---|---|
| mean | **404.4 ms** (40.4 M cycles) | 398.0 ms |
| min / median / max | 134.6 / 399.5 / 896.6 ms | 130.1 / 383.1 / 846.4 ms |
| input spikes / sample | 2396 | 2351 |
| hidden spikes / sample | 368.6 | 421.3 |
| output spikes / sample | 22.2 | 21.9 |
| SOPs / sample (in·128 + hid·10) | 310 384 | 305 191 |
| cycles / SOP (overall) | 130.3 | 130.4 |

Host overhead was 0.72 ms per sample (file read + AXI copy + mailbox). That
is ≈ 0.2 % of the total and therefore negligible.

### 5.3 Where the cycles go

A least-squares fit over the 1000 samples explains the latency almost
perfectly (R² = 0.99995):

```
cycles ≈ 121.8 · (input-layer SOPs) + 109.6 · (hidden-layer SOPs) + 2.68 M
```

- **~122 cycles per synaptic operation.** The inner loop compiles to
  7 instructions per synapse:

  ```
  664: lb   a2,0(a3)      # weight (int8)
       lw   a4,0(a5)      # cur[o]
       addi a5,a5,4
       addi a3,a3,1
       add  a4,a4,a2
       sw   a4,-4(a5)
       bne  a1,a5,664
  ```

  That is ≈ 17 cycles per instruction. PicoRV32 is not pipelined, and every
  instruction **fetch** and every **load** goes through the BRAM bus with 8
  wait states (`READ_WAIT` in `spike_soc.v`). So memory access, not
  arithmetic, dominates.
- **~2.68 M cycles (26.8 ms) fixed per sample:** bias initialisation and
  the LIF update (64-bit multiply, compare) for 138 neurons × 25 steps,
  bitmap decoding of 25 × 25 words, and state zeroing.
- **Latency is linear in the spike count.** That supports the SOP-based
  energy/cost proxy of `benchmark.py` on real hardware: on this core,
  latency ≈ 122 cycles × SOPs. The spread from 135 to 897 ms comes entirely
  from how many pixels/spikes an image has.

### 5.4 Points worth taking into the report

- The **dense ANN** of the same size needs 101 632 MACs per sample. The SNN
  needs ~310 k SOPs per sample over 25 steps, about 3× more operations. On
  this core an SOP is a load-add-store, which is not cheaper than a MAC with
  the fast multiplier. This is the core of the ANN-vs-SNN question. The ANN
  board run is still missing to measure it.
- Fewer timesteps or sparser input encoding reduce SNN latency directly
  (linear model above).
- Natural targets for the ISA-extension part: the per-synapse
  load-add-store (e.g. a packed 4×int8 add into 4 accumulators) and the LIF
  update (leak + add + compare + reset). Bus wait states are the other big
  lever, but they are a hardware change, not an ISA change.

---

## 6. Limitations / known gaps

- The ANN path is not ported yet, so there is no board ANN-vs-SNN comparison.
- Single-sample mailbox protocol. This is fine at 400 ms per sample, but
  would matter for a much faster core.
- Post-training quantization only, with per-layer scales. Per-channel
  scales or quantization-aware training were not needed (no accuracy loss).
- Energy is not measured. Only cycles and SOP counts are.
- The board clock is not NTP-synced (files are dated May), which doesn't
  matter for the measurements.

---

## 7. Raw data

- `docs/results/first_run/seed0_1000_board_results.csv`: per sample label,
  board/PC predictions, `counts_match`, cycles, spikes, output counts.
- `docs/results/first_run/seed1_100_board_results.csv`
- `docs/results/first_run/seed{0,1}_meta.json`: quantization scales,
  thresholds, PC accuracies.

The first-run CSVs predate the `in_spikes`/`hidden_spikes` columns. Their
`spikes` column is hidden + output spikes. The input spikes in §5.2 were
recomputed from `inputs.bin`.
