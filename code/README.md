# Code

| folder | what |
|---|---|
| [`ann_vs_snn_mnist/`](ann_vs_snn_mnist/README.md) | MNIST: ANN and SNN training, PC benchmark, live drawing demo |
| [`ann_vs_snn_mnist/pynq/`](ann_vs_snn_mnist/pynq/README.md) | shared board tooling for both workloads: int8 export, RISC-V firmware (all network types and firmware levels), deploy and board benchmark |
| [`ann_vs_snn_ecg/`](ann_vs_snn_ecg/README.md) | MIT-BIH ECG: preprocessing, patient-level cross-validation, feedforward and streaming models, board benchmark, live ECG demo |
| [`pynqz2_riscv_flow/`](pynqz2_riscv_flow/README.md) | PicoRV32 SoC and bring-up flow from the course, plus Group 4's SNN and MAC coprocessors (`rtl/snn_pcpi.v`, `rtl/mac_pcpi.v`), testbenches and bitstreams |
| [`snn_keyword/`](snn_keyword/README.md) | course skeleton for the other group (unchanged) |
| `overnight.sh`, `summarize_overnight.py`, `estimate_overnight.py` | accuracy-vs-cost sweeps over hidden size and timesteps ([results](../docs/results/overnight_2026-10-12/summary.md)) |

Both workloads run on the same firmware image: the model file says which network it is.
