# ANN vs SNN on a RISC-V soft core

**5XPG0, TU Eindhoven, 2026, Group 4:** Maria Mihai, Michael Scholl, Roko Ćubela,
Jan Lemieszka, Mikolaj Lapinski, Hugo Blankestijn

**Final report:** [docs/report/5XPG0_Final_Report_Group04.pdf](docs/report/5XPG0_Final_Report_Group04.pdf)

For edge inference on a small processor, is a spiking neural network (SNN) a better choice
than a classical ANN, and what hardware support makes the difference? We measured it on
real hardware rather than counting operations. **The ANN wins on dense, one-shot inputs.
The SNN wins clearly once the input is a stream:** on ECG read sample by sample, a
recurrent SNN is 107× faster than a GRU at the same accuracy, even with each network
running on its own custom coprocessor.

## Platform and method

- **Hardware:** PicoRV32 (RV32IM, 100 MHz) in the programmable logic of a PYNQ-Z2
  (Zynq-7020), from the course's SoC flow. Firmware is bare-metal C; the ARM side only
  loads inputs and reads results.
- **Workloads:** MNIST (784-128-10) and MIT-BIH ECG arrhythmia, N / S / V
  (129-64-3). ECG uses the strict inter-patient split: models are tuned with
  patient-level cross-validation and tested on 22 unseen patients.
- **Fairness rules:** both networks get the same topology and training budget, int8
  weights, the same firmware optimisations, and in the hardware comparison a coprocessor
  each.
- **Measurement:** every inference is timed in CPU cycles on the board. Every board
  output is checked against the PC int8 model; all of them match (≈ 340 000 samples).

## Design choices

| choice | why |
|---|---|
| SNN input: direct encoding for ECG (T = 10); rate coding for MNIST (T = 25) | on ECG, direct encoding was more accurate than rate coding in cross-validation (macro-F1 0.655 vs 0.639) with fewer timesteps; MNIST keeps the classic rate-coded baseline |
| ECG feature: how a beat differs from the same patient's recent beats (causal, label-free template) | fixes patients a raw-shape model gets wrong (cross-validated macro-F1 0.54 → 0.67) |
| Streaming ECG: recurrent SNN with level-crossing input vs a GRU of the same hidden size | the setting a wearable monitor really has; the SNN only does work when the signal changes |
| Firmware levels: baseline C → optimised C → custom instructions | separates what the algorithm costs from what software and hardware tuning buy |
| SNN coprocessor (`snn_pcpi.v`, 6 custom-0 instructions) | profiling showed half of the SNN's time is the per-timestep neuron update; `snn.step` updates 32 LIF neurons in one pipelined instruction |
| Matched MAC coprocessor for the ANN (`mac_pcpi.v`, 5 custom-1 instructions) | without it the hardware comparison would favour the SNN; it is built the same way and does 4 int8 MACs per instruction |
| External ANN reference: muRISCV-NN (unmodified) | shows our own ANN kernels are a strong baseline, not a strawman (13.6 ms vs 76.6 ms per MNIST image) |

Both coprocessors sit on PicoRV32's co-processor port (PCPI) and share it in one bitstream.
Each adds under 2 % of the FPGA's LUTs (SNN unit +889 LUTs and +2 DSPs; MAC unit +758 LUTs),
and together they meet timing at 100 MHz.

## Results

Latency per sample on the board. *Software* is the optimised C firmware; *hardware*
gives each network its own coprocessor.

| workload | accuracy (ANN / SNN) | software (ANN / SNN) | hardware (ANN / SNN) | verdict |
|---|---|---|---|---|
| MNIST | 96.8 % / 97.7 % | 13.6 / 57.1 ms | 5.75 / 31.9 ms | ANN, 5.5× faster |
| ECG, one beat at a time | macro-F1 0.675 / 0.706 | 4.87 / 8.30 ms | 2.13 / 2.66 ms | ANN 1.25× faster, SNN slightly more accurate |
| ECG, streamed sample by sample (GRU vs recurrent SNN) | macro-F1 ≈ 0.67 both | 1 099 / 36.4 ms | 492 / 4.58 ms | **SNN, 107× faster** |

- **One-shot inputs:** the SNN repeats its work over many timesteps, and a sparse ANN
  exploits input zeros just as well. Matched hardware narrows the ECG gap
  (1.70× → 1.25×) but does not close it, and widens the MNIST gap (4.2× → 5.5×).
- **Streaming input:** the GRU does multiply-accumulates on every sample. The recurrent
  SNN handles about 1 500 events per beat, against 1.57 M MACs for the GRU. A
  per-instruction accelerator cannot remove that difference in the amount of work.

**Limitations:**
- Energy is inferred from cycle counts; it was not measured.
- We did not try a delta-GRU (an event-driven ANN), which would narrow the streaming gap.
- The core has no cache and runs from 8-wait-state memory, so absolute times are pessimistic.

## Where to look

| | |
|---|---|
| coprocessor RTL | `code/pynqz2_riscv_flow/rtl/snn_pcpi.v`, `mac_pcpi.v`, `spike_soc.v` |
| firmware kernels | `code/ann_vs_snn_mnist/pynq/firmware/` (`main.c`: SNN and recurrent SNN, `ann.c`: ANN, `gru.c`: GRU) |
| training and int8 export | `code/ann_vs_snn_mnist/`, `code/ann_vs_snn_ecg/` |
| technical notes per step | [docs/README.md](docs/README.md) (gives a reading order) |
| raw board results | `docs/results/` (one row per sample: prediction, cycles, spike counts) |
| live demos | `code/ann_vs_snn_mnist/demo/` (draw a digit), `code/ann_vs_snn_ecg/demo/` (ECG playback); run `start_demo.bat` or `.sh`, add `--offline` to run without the board |

To rebuild: `git submodule update --init`, then follow
[code/ann_vs_snn_mnist/README.md](code/ann_vs_snn_mnist/README.md) and
[code/ann_vs_snn_ecg/README.md](code/ann_vs_snn_ecg/README.md). You need a PYNQ-Z2 with the
PYNQ 3.1 image and a RISC-V GCC (or Docker).

## Credits and license

The PicoRV32 SoC, bring-up flow and training skeletons were provided by the course
(F. Corradi, A. Naithani). PicoRV32 is by Claire Xenia Wolf (ISC). muRISCV-NN is by
TUM EDA and Arm (Apache 2.0). Everything else is the work of Group 4 and is
MIT-licensed. [LICENSE](LICENSE) gives the exact scope.
