# Raw results

Every board run writes one row per sample, with the board prediction, the
PC int8 and float predictions, the cycle count and the network activity
(`*_board_results.csv[.gz]`). It also writes a per-run summary
(`*_board_summary.json`) and the quantisation metadata of the exported
model (`*_meta.json`). File names follow
`<model>_<dataset>_seed<k>_<firmware level>`. The firmware levels are
`baseline` (straightforward C), `fast` (optimised C), `hw` (SNN coprocessor),
`mac` (SNN + MAC coprocessors) and `muriscv` (ANN on muRISCV-NN). The
`summary_*.md` files are the mean ± std tables quoted in the reports.

| folder | experiment | described in |
|---|---|---|
| `ann_vs_snn/` | MNIST ANN vs SNN, all 10 000 test images, 2 seeds, levels baseline / fast / hw / muriscv | [ann_vs_snn_pynq_comparison.md](../ann_vs_snn_pynq_comparison.md), [muriscv_reference.md](../muriscv_reference.md) |
| `ann_vs_snn_ecg/` | ECG ANN vs SNN, all 49 273 DS2 beats, 3 seeds; `pc/` PC cross-validation and benchmark; `T_study*/` SNN at other timestep counts | [ann_vs_snn_ecg_pynq_comparison.md](../ann_vs_snn_ecg_pynq_comparison.md) |
| `ecg_streaming/` | streaming ECG, GRU vs recurrent SNN; `pc/` training and CV, `board_subset/` and `board_full_hw/` board runs | [snn_coprocessor_and_streaming.md](../snn_coprocessor_and_streaming.md) |
| `snn_coprocessor/` | Vivado reports with and without the SNN unit; `v33_regression/` firmware v3.3 regression runs | [snn_coprocessor_and_streaming.md](../snn_coprocessor_and_streaming.md) |
| `mac_coproc/` | MAC unit: board runs (`board/`), RTL and whole-SoC simulation (`sim/`, `sim_rtl/`), bitstream and board regression (`regression/`), Vivado reports | [mac_coprocessor.md](../mac_coprocessor.md) |
| `overnight_2026-10-12/` | supplementary sweeps of hidden size and T (`summary.md`) | [../README.md](../README.md) |
| `full_testset/` | first full-test-set MNIST SNN run (development log) | [snn_mnist_pynq_full_testset.md](../snn_mnist_pynq_full_testset.md) |
| `first_run/`, `full_run_partial/` | first MNIST SNN board runs, including an interrupted one (development log) | [snn_mnist_pynq_deployment.md](../snn_mnist_pynq_deployment.md), [snn_mnist_pynq_full_testset.md](../snn_mnist_pynq_full_testset.md) |

Every number in the final report is collected from these files by
[`../report_analysis/compute_results.py`](../report_analysis/compute_results.py)
into `../report_analysis/results_summary.md`.
