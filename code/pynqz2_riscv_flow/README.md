# PYNQ-Z2 RISC-V + SNN bring-up flow

A minimal, self-contained starting point for both project groups. It puts a
**PicoRV32** RISC-V soft core on the PYNQ-Z2 FPGA, feeds it with a **hardware
Poisson spike generator**, and runs a **single LIF neuron in C** that emits
output spikes the board can read.

Everything a last-year EE student needs is here; nothing is solved for you.

```
rtl/       picorv32.v (core, vendored)  spike_soc.v  poisson.v  ps_if.v  spike_top.v
firmware/  main.c (the demo neuron)     board.h     hal.h     start.S   linker.ld
vivado/    build.tcl  spike_top.xdc  spike_top.bit (prebuilt)  board_files/
host/      spike_pynq.py (load/run/read)  demo.sh  smoke_test.sh
install.sh  run_demo.sh  Makefile  requirements.txt
```

## What the design does

```
  PC / Linux (PS)                    FPGA fabric (PL)
  ┌───────────────┐   AXI-Lite      ┌──────────────────────────────┐
  │ spike_pynq.py │ ───────────────▶│ ps_if ─ BRAM (256 KB)        │
  │  load bit/elf │                 │        │                     │
  │  read console │                 │     PicoRV32 ── firmware     │
  │  read spikes  │◀─────────────── │        │                     │
  └───────────────┘                 │   poisson  LED  timer  sys  │
                                    └──────────────────────────────┘
```

* The **PS** loads the bitstream and the firmware image into the PL over AXI,
  then reads back the console and the output-spike log.
* The **Poisson generator** (`rtl/poisson.v`) produces input spikes on 8
  independent channels at programmable rates.
* The **firmware** (`firmware/main.c`) runs one leaky integrate-and-fire
  neuron, integrates the spikes, and logs an output spike + flashes the LEDs
  when it fires.

## Prerequisites (once per board)

* A **PYNQ-Z2** with the stock **PYNQ 3.1** SD image and an Ethernet cable.
* **SSH key auth** for user `xilinx` (`ssh xilinx@<board-ip>` must work with
  no password). On the stock image the sudo password is `xilinx`
  (`user xilinx`, `password xilinx`) and it is **passwordless-friendly**.
* PC and board on the same network; the board can reach the internet.

## Quick start

```bash
# 1. deploy (this also builds nothing - the prebuilt bitstream is included)
./install.sh 192.168.2.99

# 2. run the demo (loads bitstream + firmware, prints banner + spikes)
./run_demo.sh 192.168.2.99
```

Or with `make`:

```bash
make install BOARD_IP=192.168.2.99
make demo    BOARD_IP=192.168.2.99
make smoke   BOARD_IP=192.168.2.99     # install + self-test
```

If you prefer to do it by hand on the board (files land in `/home/xilinx/snn`):

```bash
PY=/usr/local/share/pynq-venv/bin/python3
sudo bash -c "source /etc/profile.d/xrt_setup.sh && $PY spike_pynq.py load-bit spike_top.bit"
sudo bash -c "source /etc/profile.d/xrt_setup.sh && $PY spike_pynq.py load-elf spike.bin"
sudo bash -c "source /etc/profile.d/xrt_setup.sh && $PY spike_pynq.py start"
sudo bash -c "source /etc/profile.d/xrt_setup.sh && $PY spike_pynq.py console"
sudo bash -c "source /etc/profile.d/xrt_setup.sh && $PY spike_pynq.py spikes"
```

## Rebuilding

```bash
make firmware                 # RISC-V toolchain (riscv64-unknown-elf-gcc)
make bitstream                # Vivado 2024.1, ~15 min
```

The prebuilt `vivado/spike_top.bit` is a reference; you can always rebuild it
from the RTL. The Vivado version matters: the **Zynq-7000 family is only in
Vivado 2024.1** (`vivado/vivado2024.sh` selects it).

## Make it yours

Places to start (they are all commented in the source):

| Goal                         | Where                                             |
|------------------------------|---------------------------------------------------|
| Change input encoding        | `rtl/poisson.v`, `firmware/main.c` (rates)        |
| Change the neuron/network    | `firmware/main.c` (the LIF + weights)             |
| Add output neurons / layers  | `firmware/main.c`, `board.h` (`NCH`)              |
| Add a real input path (PC→PL)| `host/spike_pynq.py` mailbox + a new BRAM buffer  |
| Add ISA extensions           | `picorv32` PCPI hooks (group 4 extra)             |

## Memory map (needed for your own code)

RISC-V view: `0x0000_0000` BRAM, `0x1000_0000` sysctrl,
`0x1000_1000` timer, `0x1000_2000` LED, `0x1000_3000` Poisson.
PS/AXI view: `0x4000_0000` BRAM, `0x4004_0000` syscon.
See `firmware/board.h` and `host/spike_pynq.py` (they must stay in sync).

## Added by Group 4: SNN and MAC coprocessors

Two custom-instruction units on the PicoRV32 PCPI port. Each is enabled by a Verilog
define that the build script sets. Without the defines, `rtl/spike_soc.v` builds the
course SoC unchanged; we checked this by rebuilding and comparing the bitstream and the
board results.

| | SNN unit | MAC unit |
|---|---|---|
| RTL | `rtl/snn_pcpi.v` (define `SNN_COPROC`) | `rtl/mac_pcpi.v` (define `MAC_COPROC`) |
| instructions | 6, opcode custom-0: LIF update of up to 32 neurons per instruction, 4-weight current update, ... | 5, opcode custom-1: 4 int8 MACs per instruction into 256 accumulators |
| cost over the SoC | +889 LUTs, +342 FFs, +2 DSPs | +758 LUTs, +188 FFs, 0 DSPs |
| testbench | `sim/tb_snn_pcpi.v` (vectors from `sim/gen_vectors.py`) | `sim/tb_mac_pcpi.v` (`sim/gen_vectors_mac.py`) |

- **Bitstreams:** `vivado/build_snn.tcl` builds `spike_top_snn.bit` (SNN unit) and
  `vivado/build_both.tcl` builds `spike_top_both.bit` (both units, timing met at
  100 MHz). They were built with Vivado 2026.1 on our PC.
- **Whole-SoC simulation without a board:** `sim/tb_soc.v`, driven by `sim/soc_sim.py`.
- **Firmware** that uses the units is in `../ann_vs_snn_mnist/pynq/firmware/`.
- **Design and measurements:** `docs/snn_coprocessor_and_streaming.md` and
  `docs/mac_coprocessor.md`.
