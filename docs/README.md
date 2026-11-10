# Documentation: reading guide

The final report is the condensed version of everything here:
[report/5XPG0_Final_Report_Group04.pdf](report/5XPG0_Final_Report_Group04.pdf).
The documents below are the detailed technical record behind it: design
decisions, measurements and checks for each step.

## Suggested reading order

| # | document | what it answers |
|---|---|---|
| 1 | [ann_vs_snn_pynq_comparison.md](ann_vs_snn_pynq_comparison.md) | MNIST: ANN vs SNN on the PicoRV32, baseline and optimised firmware, where the cycles go |
| 2 | [ann_vs_snn_ecg_pynq_comparison.md](ann_vs_snn_ecg_pynq_comparison.md) | ECG (MIT-BIH, inter-patient): preprocessing, cross-validation, board results, timestep study |
| 3 | [snn_coprocessor_and_streaming.md](snn_coprocessor_and_streaming.md) | SNN neuron coprocessor (custom-0 instructions); streaming ECG with a recurrent SNN vs a GRU |
| 4 | [mac_coprocessor.md](mac_coprocessor.md) | matched MAC coprocessor for the ANN (custom-1), so both sides get hardware support; final hardware comparison |
| 5 | [muriscv_reference.md](muriscv_reference.md) | external ANN reference: the same int8 ANN on the unmodified muRISCV-NN library |

## Development logs

Written during bring-up. They are kept for traceability; their results are
superseded by documents 1–5.

- [snn_mnist_pynq_deployment.md](snn_mnist_pynq_deployment.md): first MNIST SNN on the board: setup, quantisation, input path, first (baseline) run.
- [snn_mnist_pynq_full_testset.md](snn_mnist_pynq_full_testset.md): optimised firmware and the first full-test-set run.

## Data and figures

- [results/](results/README.md): raw per-sample board results, summaries and
  Vivado reports, one folder per experiment.
- [report_analysis/](report_analysis/): `compute_results.py` collects every
  number used in the report into `results_summary.md` / `.csv` / `results.json`;
  `make_report_figures.py` and `make_slide_figures.py` draw the report and
  presentation figures from it (`figures_report/`, `figures/`).
- [results/overnight_2026-10-12/](results/overnight_2026-10-12/summary.md):
  supplementary accuracy-vs-cost sweeps (hidden size and number of timesteps
  T), produced by `code/overnight.sh` and `code/summarize_overnight.py`.

## Course material

- [course/README.md](course/README.md): the original 5XPG0 project description.
