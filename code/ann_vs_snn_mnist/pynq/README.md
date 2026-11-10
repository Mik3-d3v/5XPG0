# Board tooling: run the networks on the PYNQ-Z2 RISC-V core

Bare-metal RV32IM C on PicoRV32, built on the crt0, linker script and host tool of
`../../pynqz2_riscv_flow`. One firmware image runs every network type (the magic of
`model.bin` selects it): ANN, SNN, GRU and recurrent SNN. The same tooling also runs
the ECG models (exporter and benchmark in `../../ann_vs_snn_ecg/pynq/`). Every level
gives bit-identical outputs; only the speed differs.

| level (`--fw`) | image / bitstream | what changes | MNIST SNN | MNIST ANN |
|---|---|---|---|---|
| `baseline` | `snn_mnist.bin` / `spike_top.bit` | straightforward C, byte by byte | 403 ms | 136 ms |
| `fast` (default) | `snn_mnist_fast.bin` / `spike_top.bit` | delta currents, nonzero inputs only, 4×int8 SWAR | 57.1 ms | 13.6 ms |
| `muriscv` | `snn_mnist_muriscv.bin` / `spike_top.bit` | ANN on the unmodified muRISCV-NN library (reference) | 57.1 ms | 76.6 ms |
| `hw` | `snn_mnist_hw.bin` / `spike_top_snn.bit` | SNN on the SNN coprocessor (custom-0) | 31.9 ms | 13.6 ms |
| `mac` | `snn_mnist_mac.bin` / `spike_top_both.bit` | plus the ANN on the MAC coprocessor (custom-1) | 31.9 ms | 5.75 ms |

Design, decisions and results (repo root):
- `docs/ann_vs_snn_pynq_comparison.md` (MNIST results)
- `docs/snn_coprocessor_and_streaming.md`, `docs/mac_coprocessor.md` (coprocessor levels)
- `docs/snn_mnist_pynq_deployment.md`, `docs/snn_mnist_pynq_full_testset.md` (bring-up logs)

```
benchmark_pynq.py      PC: benchmark.py on the board (ANN + SNN, all seeds, mean +- std)
export_model.py        PC: quantize checkpoints, encode test inputs, int reference
build_fw.sh            PC: build all firmware images (local riscv gcc or Docker)
deploy_pynq.sh         PC: copy everything to the board and run it over SSH
status_pynq.sh         PC: progress of a detached board run, copy results back
firmware/main.c        RISC-V: SNN (integer LIF layers), mailbox, dispatch
firmware/ann.c         RISC-V: ANN (int8 Linear + ReLU + uint8 requantization)
firmware/snn_layout.h  BRAM layout + model header (mirrored in the Python scripts)
host/run_mnist_pynq.py board: load bit + fw + model, stream inputs, compare
```

## Benchmark on the board (Git Bash, from `code/ann_vs_snn_mnist/`)

```bash
source .venv/Scripts/activate          # after train_mnist.py has run
python pynq/benchmark_pynq.py          # ANN + SNN, every seed, full test set (~25 min)
python pynq/benchmark_pynq.py --n 1000 # quicker
python pynq/benchmark_pynq.py --models ANN
python pynq/benchmark_pynq.py --fw baseline --n 1000
```

This exports each model of `runs/manifest.json` (if needed), runs it on the
board, and writes `runs/summary_pynq_<fw>.md` / `.json`. These hold an
ANN-vs-SNN table with the metrics of `benchmark.py` plus board latency,
executed operations and bit-exactness, as mean ± std over seeds.

## Single runs

```bash
python pynq/export_model.py            # every model in runs/manifest.json, 1000 test images
pynq/build_fw.sh                       # only after changing the firmware
pynq/deploy_pynq.sh 192.168.2.99 snn_mnist_seed0             # all exported samples
pynq/deploy_pynq.sh 192.168.2.99 snn_mnist_seed1 -- --n 100  # first 100 only
pynq/deploy_pynq.sh 192.168.2.99 snn_mnist_seed0 -- --fw baseline --n 50
pynq/deploy_pynq.sh 192.168.2.99 ann_mnist_seed0             # the ANN, same way
```

The board prerequisites are the same as for `pynqz2_riscv_flow/install.sh`
(SSH key and passwordless sudo). Results are printed and saved to
`export/<name>/board_results_<fw>.csv` (one row per sample) and
`board_summary_<fw>.json`.

### Long runs in the background

```bash
python pynq/export_model.py --n-samples 10000                        # SNN 25 MB, ANN 7.8 MB inputs
pynq/deploy_pynq.sh 192.168.2.99 snn_mnist_seed0 --detach            # runs on the board, returns
pynq/status_pynq.sh 192.168.2.99 snn_mnist_seed0                     # progress + copy results back
pynq/status_pynq.sh 192.168.2.99 snn_mnist_seed0 -f                  # follow until done
pynq/deploy_pynq.sh 192.168.2.99 snn_mnist_seed0 --detach -- --resume  # after an interruption
```

`--detach` runs the job with `nohup` on the board, so closing the PC or the
SSH session does not stop it. Results are appended after every sample and can
be resumed. Files are only uploaded when their md5 differs from the copy on the
board. Only one run can be active at a time, because they share the FPGA.

## Why the weights are quantized

The float32 weights (784·128 + 128·10 ≈ 101 k parameters, ~400 KB) do not
fit in the 256 KB BRAM, and PicoRV32 has no FPU. Each layer is therefore
quantized with a single scale `s = max|W| / 127`:

* weights are int8, biases are int32, and `thr_q = round(thr / s)`;
* the membrane is an int32 in units of `s`;
* `beta` is a Q16 integer.

Spikes are binary, so the scales of different layers never interact. The
integer LIF matches snntorch `Leaky` (reset by subtraction, `reset_delay=True`):

```
mem_t = ((beta_q16 * mem_{t-1}) >> 16) + b + sum(W[active inputs]) - spk_{t-1} * thr_q
spk_t = mem_t > thr_q             prediction = first argmax of the output spike counts
```

The ANN uses the same int8 weight scheme. Its inputs are the uint8 pixels
(exact, because `ToTensor` = pixel / 255). Hidden activations are
`min(255, (relu(acc) * m_q + 2^23) >> 24)` (uint8), with the activation scale
calibrated on 10 000 **training** images. The logits are int32.

`export_model.py` also runs both integer models on the PC
(`int_forward_snn`, `int_forward_ann`). The board has to reproduce their
outputs (spike counts / logits) **exactly**, which makes `board == PC int ref`
the correctness check of the port.

## Input path

The SNN inputs are encoded exactly as in `benchmark.py`: `rate_encode` on the
test set, batches of 256, and `torch.manual_seed(seed)` of the checkpoint.
This gives the board the same spike trains that the software benchmark used.
Each sample is stored as `[T][25]` uint32 bitmaps (784 bits per step,
2500 B per sample). The ANN gets the 784 raw pixels (784 B per sample) of
the same test images. Per sample, the PS:

1. writes the sample to `INPUT_BASE`,
2. sends mailbox command `MB_CMD_RUN(T)`,
3. reads `pred / cycles / spikes` from the reply and the outputs (spike
   counts or logits) from `RESULT_BASE`.

The Poisson generator of the bring-up flow is not used.

## Memory map (BRAM, RISC-V addresses = offsets from PS 0x4000_0000)

| range               | content                                      |
|---------------------|----------------------------------------------|
| 0x00000 – 0x0FFFF   | firmware (≤ 6 KB code + ≤ 55 KB bss)         |
| 0x10000 / 0x10400   | console / mailbox (same as the flow)         |
| 0x10800             | SNN output spike counts / ANN logits         |
| 0x11000 – 0x12FFF   | input of one sample                          |
| 0x13000 – 0x3EFFF   | model.bin (176 KB room → hidden ≤ ~220 for 784-H-10) |
| 0x3F000 – 0x3FFFF   | stack                                        |

The firmware reads the layer count and sizes from the `model.bin` header
(up to 4 layers with at most 1024 neurons each: `fc<i>`/`lif<i>` for the
SNN, the `Linear` layers of `nn.Sequential` for the ANN). Other
widths or depths therefore only need a re-export, not a firmware rebuild.

## Limits of the platform

* Every instruction fetch goes through an 8-wait-state BRAM bus (`READ_WAIT` in
  `rtl/spike_soc.v`) and PicoRV32 has no cache, so absolute times are pessimistic.
  Both networks pay this equally.
