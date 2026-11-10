# ANN vs SNN on MIT-BIH ECG (N / S / V), PC and PYNQ-Z2 RISC-V

The ECG counterpart of [`../ann_vs_snn_mnist`](../ann_vs_snn_mnist): a 2-layer
ANN and a 2-layer SNN (129-64-3), trained under the same budget, tuned with
patient-level cross-validation, quantised to int8 and run **bit-exact** on
the PicoRV32 RISC-V core of the PYNQ-Z2 with the **same firmware image** as
MNIST.

Report: [`docs/ann_vs_snn_ecg_pynq_comparison.md`](../../docs/ann_vs_snn_ecg_pynq_comparison.md) ·
raw results: [`docs/results/ann_vs_snn_ecg/`](../../docs/results/ann_vs_snn_ecg/)

## Protocol

* **Data:** MIT-BIH Arrhythmia DB, lead MLII, AAMI classes N / S / V (F and Q
  dropped, the 4 paced records excluded), de Chazal inter-patient split:
  **DS1** (22 records) for training and tuning, **DS2** (22 other patients)
  for testing. No patient is ever in both.
* **Tuning only by patient-level 5-fold CV on DS1** (pooled out-of-fold
  predictions). DS2 is touched once per final configuration.
* **Primary metric: macro-F1 over N/S/V**, plus accuracy, per-class Se and
  +P, confusion matrices and per-record tables. Accuracy alone is
  meaningless (N is ~90 % of the beats).
* 3 seeds for everything, mean ± std; paired per-beat comparisons.

## Pipeline

| step | file | what |
|---|---|---|
| 1 | `prepare_mitbih.py` | band-pass 0.5–40 Hz, 360 → 180 Hz, 126-sample window around each R peak, per-beat z-score, RR features, **causal per-record template** (median of the previous 32 beats, labels unused) |
| 2 | `train_ecg.py` | ANN and SNN (snnTorch LIF), equal budget; CV mode (default), `--final` |
| 3 | `summarize_cv.py` | one table over all CV variants (`runs/cv/<tag>.json`) |
| 4 | `benchmark_ecg.py` | PC benchmark on DS2: metrics, per-record and paired tables |
| 5 | `pynq/export_ecg.py` | int8 export (shared `quant.py`), input scale calibrated on DS1 |
| 6 | `pynq/benchmark_pynq_ecg.py` | runs every model on the board, ECG metrics from the board predictions |

Model input (129 features): `beat − template` (126, `--morph diff`) and 3
RR features (pre-RR and post-RR relative to the recent mean, and the mean).
The SNN uses direct encoding: the feature vector is a constant input
current, T = 10 steps, β = 0.9; the class is the output neuron with the
most spikes.

## Reproduce

Git Bash, from this folder:

```bash
./setup_venv.sh && source .venv/Scripts/activate      # .venv/bin/activate on Linux
python prepare_mitbih.py                              # downloads MIT-BIH (~100 MB)

# tuning: patient-level CV on DS1 (the chosen configuration)
python train_ecg.py --use-rr --morph diff --augment --epochs 10 --weight-decay 0.2 \
    --timesteps 10 --beta 0.9 --encoding direct --tag chosen
python summarize_cv.py                                # all variants -> runs/cv/ALL.md

# final models on all of DS1, one PC benchmark on DS2
python train_ecg.py --final --use-rr --morph diff --augment --epochs 10 \
    --weight-decay 0.2 --timesteps 10 --beta 0.9 --encoding direct
python benchmark_ecg.py

# on the board (PYNQ-Z2 reachable over SSH, see ../ann_vs_snn_mnist/pynq/README.md)
../ann_vs_snn_mnist/pynq/build_fw.sh                  # only after firmware changes
python pynq/benchmark_pynq_ecg.py                     # fast firmware, all DS2 beats
python pynq/benchmark_pynq_ecg.py --fw baseline
python pynq/benchmark_pynq_ecg.py --fw mac              # both coprocessors
python pynq/benchmark_pynq_ecg.py --stream             # streaming: GRU vs recurrent SNN
python ../ann_vs_snn_mnist/pynq/compare_pynq.py --task ecg \
    --export-root pynq/export --out runs/compare_pynq.md
```

All tuning variants that were tried, with their CV results, are in
`docs/results/ann_vs_snn_ecg/pc/cv/`. The sweep commands are of the form
`python train_ecg.py --use-rr --seeds 0 1 2 <options> --tag <name>`; the
options of each variant are stored in its JSON (`args`).

## Live demo

`demo/start_demo.bat` (double-click) or `demo/start_demo.sh [--offline]`
opens http://localhost:8001: a held-out patient's ECG plays back at 0.5x to
25x while two networks classify every beat on the PYNQ-Z2 (ANN vs SNN, or
GRU vs recurrent SNN; the SNN on software or on the custom instructions).
Every board result is checked against the PC int8 model. Software mode
(`--offline` or the switch in the page) runs the PC int8 models only. Keys:
space play/pause, S / V jump to the next S / V beat, B / G model pair,
1 / 2 SNN firmware.
