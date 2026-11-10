# A matched custom-instruction unit for the ANN: MAC coprocessor

*Measured 2026-10-14. Follows [snn_coprocessor_and_streaming.md](snn_coprocessor_and_streaming.md),
whose SNN unit (`snn_pcpi.v`) had no ANN-side counterpart. This adds one, built the same way, so
that the hardware comparison has something on both sides. Software-vs-software results are unchanged.*

### Headline (board, PicoRV32 @ 100 MHz, all outputs bit-exact against the PC int8 models)

| setting | ANN sw | ANN + MAC unit | SNN sw | SNN + SNN unit (+ MAC 1st layer) |
|---|---|---|---|---|
| MNIST (2 seeds × 10 000) | 13.62 ms | **5.75 ms** | 57.13 ms | **31.87 ms** ¹ |
| ECG beat (3 seeds × 49 273) | 4.87 ms | **2.13 ms** | 8.30 ms | **2.66 ms** (SNN unit only: 5.32) |
| ECG stream, GRU / RSNN (3 seeds × 1 000) | 1 099 ms | **492 ms** | 36.4 ms | **4.58 ms** ² |

¹ The MNIST SNN is rate-coded (spike inputs), so it has no MAC layer: same image path and same
cycles as on the SNN unit alone. ² The RSNN has event inputs and no dense layer; it does not use
the MAC unit (same cycles as before).

| same sample, same seed | SNN faster before (SNN unit vs ANN sw) | SNN faster now (both units vs ANN + MAC) |
|---|---|---|
| MNIST | 2 / 20 000 | **0 / 20 000** |
| ECG beat | 884 / 147 819 | **6 / 147 819** |
| ECG stream (RSNN vs GRU) | 3 000 / 3 000 | **3 000 / 3 000** |

Accuracy does not move (bit-exact): MNIST 96.84 / 97.65 %, ECG macro-F1 0.675 / 0.706, stream
macro-F1 0.664 / 0.655 (ANN side / SNN side, on the beats run).

**Conclusion.** A matched MAC unit makes the ANN 2.3–2.4× faster and the GRU 2.2×. In the
feedforward settings that puts the ANN *further* ahead: SNN/ANN goes from 1.09 to 1.25 (ECG)
and from 2.34 to 5.5 (MNIST), and the SNN wins 6 of 147 819 beats instead of 884. In the streaming
setting the advantage **survives a matched ANN extension**: the RSNN stays **107× faster** than
the GRU on its own MAC unit (240× before), on every beat. The GRU still has 1.57 M MACs per beat
to issue, the RSNN about 1 500 synaptic events.

---

## 1. Design

`code/pynqz2_riscv_flow/rtl/mac_pcpi.v`, on PicoRV32's PCPI port like `snn_pcpi.v`, built with
the Verilog define `MAC_COPROC`. All R-type, opcode **custom-1** (`0101011`), op in funct7.

| funct7 | instruction | operands | effect | cycles |
|---|---|---|---|---|
| 0x00 | `mac.clr` | first, count | accumulators = 0 | count |
| 0x01 | `mac.setx` | x | X = x (low 9 bits, signed: int8 or uint8) | 1 |
| 0x02 | `mac.acc4` | neuron (×4), 4 packed int8 weights; f3[0] = offset binary | acc[n+k] += X · w_k, k = 0..3 | 1 |
| 0x03 | `mac.get` | neuron | rd = acc[n] | 1 |
| 0x04 | `mac.setacc` | neuron, value | acc[n] = value (bias) | 1 |

- 256 int32 accumulators in 4 banks of distributed (LUT) RAM, like the SNN unit's `cur0..cur3`,
  so `acc4` updates 4 outputs per cycle. Same handshake and state machine (`S_IDLE/S_RUN/S_DONE`).
- `mac.acc4` is the analogue of `snn.acc4`. There, a spiking input adds ±1 × its weight row to
  the currents; here, the input value X scales the row. The one structural difference is that the
  four 9 × 8-bit products are registered when the instruction is accepted, so the run cycle only
  adds. Vivado built the multipliers from LUTs (0 DSPs).
- **SoC** (`rtl/spike_soc.v`): `MAC_COPROC` alone gives a MAC-only SoC. `SNN_COPROC` + `MAC_COPROC`
  puts both units on the one PCPI port. Each unit claims only its own opcode, so `wr / wait /
  ready` are ORed and `rd` comes from the unit that is ready. `vivado/build_both.tcl` →
  `spike_top_both.bit` (board image); `vivado/build_mac.tcl` → `spike_top_mac.bit` (cost only).
- While checking that the default build is unchanged, a **pre-existing bug** turned up. The
  no-define branch added with the SNN unit (`wire pcpi_valid, pcpi_wr = 1'b0, ...`) never
  synthesized in Vivado 2026.1 ("all identifiers need to have initial value"). The shipped
  `spike_top.bit` predates it. The declarations are now split; see §3.2 for the result.

### Firmware

`ann.c`, level `FW_MAC` (fast build + MAC unit). Only the inner loop of the fast `mac_layer()` is
replaced. It keeps the sparse input scan (zero words skipped, nonzero inputs only); per nonzero
input it issues `mac.setx x` and then one `mac.acc4` per 4 weights of its row (`.insn r 0x2B ...`,
f3[0] = 1 since the fast build keeps weights in offset binary). The bias is loaded with
`mac.setacc` and results are read with `mac.get`. The unit undoes the +128 offset itself, so the
`128·Σx` correction is gone. The inner loop is 5 instructions per 4 MACs (load, `acc4`, 2 ×
increment, branch), the same shape as the SNN unit's `rows_hw`. Layers wider than 256 outputs fall
back to the software kernel (none in this study).

Image `snn_mnist_mac.bin` (`--fw mac`, `spike_top_both.bit`) = the hw image + this `mac_layer()`:

- ANN: every layer on the MAC unit;
- GRU: `Wh h` (3H = 192 outputs) on the MAC unit, gates in software as before;
- direct-encoded SNN (ECG): first layer (one MAC pass) on the MAC unit, timesteps on the SNN unit,
  i.e. **both** units;
- rate-coded SNN (MNIST) and RSNN: SNN unit only (they have no MAC layer).

Rebuilding all firmware gives byte-identical `snn_mnist{,_fast,_hw,_muriscv}.bin`.

## 2. Verification

- **Unit testbench** (`sim/tb_mac_pcpi.v`, `sim/gen_vectors_mac.py`, xsim 2026.1). A Python model
  of the instructions generates a random stream of 4 257 instructions. The stream includes
  int8/uint8/edge X values, biases near the int32 wrap-around, garbage in unused operand bits, and
  361 *foreign* instructions (custom-0 SNN opcodes, unused funct7) that the unit must not claim.
  The testbench drives them through the PCPI handshake as the CPU does: **977 / 977 results
  correct, 0 errors**. Two deliberate mutations are caught: dropping the offset-binary correction
  (225 errors) and treating X as unsigned (207 errors). Logs: `docs/results/mac_coproc/sim/`.
- **Whole SoC in simulation** (`sim/tb_soc.v`, `sim/soc_sim.py`): PicoRV32, BRAM with its wait
  states and both units, running the real firmware images on real exported samples. It was built
  while the board was briefly unreachable. It turned out to be **cycle-exact**: on every one of
  the 2 630 samples simulated with the MAC image, the board later gave identical outputs and
  identical cycles (`regression/board_vs_sim_mac.txt`). This makes it a usable pre-board check
  for future firmware levels.
- **Board**: every output of every run equals the PC int8 reference (MNIST 20 000 ANN + 20 000
  SNN, ECG 147 819 + 147 819, stream 3 000 + 3 000).

## 3. Cost and regression

### 3.1 Cost (Vivado 2026.1, routed, 100 MHz)

| | stock | SNN unit | MAC unit | both units |
|---|---|---|---|---|
| Slice LUTs | 2 631 | 3 520 (+889) | 3 389 (+758) | 4 301 (+1 670) |
| of which LUT RAM | 104 | 457 | 232 | 585 |
| registers | 2 550 | 2 892 (+342) | 2 738 (+188) | 3 051 (+501) |
| DSPs | 6 | 8 (+2) | 6 (+0) | 8 (+2) |
| timing (WNS) | +0.632 ns | +0.447 ns | +0.667 ns | **+0.803 ns, met** |
| on-chip power (estimate) | 1.682 W | 1.695 W | 1.685 W | 1.691 W |

Per unit in the both-units build (hierarchical report): SNN unit 913 LUTs (353 LUT RAM),
339 registers, 2 DSPs, 10 mW; **MAC unit 606 LUTs (128 LUT RAM), 171 registers, 0 DSPs, 3 mW**.
The remaining LUTs are the CPU's PCPI decode, which either unit enables. The MAC unit is the
smaller of the two, so the comparison does not favour the SNN by giving the ANN less hardware.
Reports: `docs/results/mac_coproc/vivado_both/`, `vivado_mac/`.

### 3.2 Default builds unchanged

- **Bitstreams** (`regression/bitstream_identity.txt`), rebuilt from the modified RTL and
  compared byte by byte. SNN unit only (`build_snn.tcl`) vs the shipped `spike_top_snn.bit`:
  configuration data identical. No define (`build.tcl`, fixed declarations) vs the 2026.1
  reference build of 2026-10-02 (board-identical to the released `spike_top.bit`): configuration
  data identical. In both cases only the build date/time in the header differs.
- **Board** (`regression/board_regression.txt`): stock bitstream + fast firmware and SNN-only
  bitstream + hw firmware, 2 000 ECG beats and 200 MNIST images each, vs the published v3.3 board
  runs: **4 400 / 4 400 samples with identical outputs and identical cycle counts**.
- The ANN of the earlier levels and the SNN-unit levels are therefore unaffected.

## 4. Results in detail (board, mean ± std over seeds)

| | ANN fast | ANN mac | ANN speed-up | SNN hw | SNN mac | SNN speed-up | SNN/ANN before → now |
|---|---|---|---|---|---|---|---|
| MNIST | 13.62 ms | 5.749 ± 0.000 ms | 2.37× | 31.87 ms | 31.874 ± 0.089 ms | 1.00× | 2.34 → **5.55** |
| ECG | 4.872 ms | 2.134 ± 0.007 ms | 2.28× | 5.316 ms | 2.658 ± 0.030 ms | 2.00× | 1.09 → **1.25** |

| ECG stream (1 000 beats × 3 seeds) | GRU | RSNN | GRU / RSNN |
|---|---|---|---|
| software (fast) | 1 099 ms | 36.4 ms | 30× |
| SNN unit (hw) | 1 100 ms | 4.58 ms | 240× |
| **matched units (mac)** | **492.5 ± 0.4 ms** | **4.58 ± 0.03 ms** | **107×** |

- **Why only ~2.3×.** The unit does 4 MACs per cycle, but the core still issues one instruction
  per 4 weights plus a load and loop bookkeeping, each fetched through the 8-wait-state BRAM. The
  SNN unit's `snn.acc4` has exactly the same limit, so both sides get the same kind of help.
- **ECG SNN with both units** (2.66 ms) = the first layer, now as fast as the ANN's, plus T = 10
  steps on the SNN unit (~0.5 ms). The SNN can at best tie the ANN; it is faster on 6 beats.
- **GRU.** `Wh h` per step is up to 64 nonzero inputs × 48 `acc4`. At 126 steps per beat, a beat
  still takes 0.49 s: not real time (> 55 bpm), and two orders of magnitude above the RSNN.

Raw data: `docs/results/mac_coproc/board/` (per-sample CSVs and summaries in the format of the
earlier runs; `*_regr` = regression runs); RTL simulation subsets in `sim_rtl/` with
`summary_sim_rtl.md`. The rows `*/mac` are in `docs/report_analysis/results.json` /
`results_summary.*` (`compute_results.py`).

## 5. Interpretation

1. **Feedforward: matched hardware helps the ANN more.** Each side now has its own small unit.
   The ANN's whole workload is MACs. The SNN's direct-encoded first layer is the same MAC pass,
   and its timesteps come on top. So "the hardware worth adding is the neuron update" holds only
   against an *unaided* ANN. With matched units the snapshot SNN is behind again (ECG 1.25×,
   faster on 6 / 147 819 beats).
2. **Streaming: the advantage survives.** With a matched unit the GRU drops from 1.1 s to 0.49 s,
   but the RSNN on its unit needs 4.6 ms: 107× instead of 240×, on every beat. The gap comes from
   the amount of work (MACs every sample vs a few events), which no per-instruction accelerator
   removes.
3. **Cost is comparable**: MAC unit 606 LUTs / 0 DSPs / 3 mW, SNN unit 913 LUTs / 2 DSPs / 10 mW.
   Both together meet 100 MHz.

## 6. Reproduce

```bash
# unit testbench (code/pynqz2_riscv_flow/sim/; xsim needs a path without spaces)
python gen_vectors_mac.py && xvlog ../rtl/mac_pcpi.v tb_mac_pcpi.v && xelab tb -s tb && xsim tb -R
# bitstreams (code/pynqz2_riscv_flow/vivado/)
vivado -mode batch -source build_both.tcl      # -> spike_top_both.bit
vivado -mode batch -source build_mac.tcl       # -> spike_top_mac.bit (cost table only)
# firmware (code/ann_vs_snn_mnist/)
pynq/build_fw.sh                               # + snn_mnist_mac.bin
# board (deploy_pynq.sh uploads spike_top_both.bit and snn_mnist_mac.bin)
python pynq/benchmark_pynq.py --fw mac                       # code/ann_vs_snn_mnist/
python pynq/benchmark_pynq_ecg.py --fw mac                   # code/ann_vs_snn_ecg/
python pynq/benchmark_pynq_ecg.py --stream --fw mac
# regression without touching the published CSVs (--tag)
bash ../ann_vs_snn_mnist/pynq/deploy_pynq.sh 192.168.2.99 $(pwd)/pynq/export/ann_ecg_seed0 \
     -- --fw fast --n 2000 --tag _regr
# whole-SoC simulation (no board), e.g. ECG ANN on the MAC unit, 200 random beats
python code/pynqz2_riscv_flow/sim/soc_sim.py --fw code/ann_vs_snn_mnist/pynq/firmware/snn_mnist_mac.bin \
       --defines SNN_COPROC MAC_COPROC --export code/ann_vs_snn_ecg/pynq/export/ann_ecg_seed0 --random 200
python docs/report_analysis/mac_coproc_summary.py && python docs/report_analysis/compute_results.py
```
