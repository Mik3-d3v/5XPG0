# ANN vs SNN on MNIST, PC and PYNQ-Z2 RISC-V

A 2-layer ANN and a 2-layer SNN (784-128-10), trained under the same budget, quantised to
int8 and run **bit-exact** on the PicoRV32 RISC-V core of the PYNQ-Z2. The same firmware
image also runs the [ECG models](../ann_vs_snn_ecg/README.md).

Report: [`docs/ann_vs_snn_pynq_comparison.md`](../../docs/ann_vs_snn_pynq_comparison.md) ·
raw results: [`docs/results/ann_vs_snn/`](../../docs/results/ann_vs_snn/)

## Protocol

* **Same budget for both networks:** topology, optimiser, learning-rate schedule, epochs,
  initialisation seed, data order and train/validation split.
* **Data:** 5 000 training images held out for validation; the test set is used only by
  the benchmarks.
* **SNN:** snnTorch LIF neurons, rate-coded input with T = 25 steps, β = 0.9; the class is
  the output neuron with the most spikes.
* **Reporting:** 2 seeds, mean ± std. Board results use all 10 000 test images.

## Pipeline

| step | file | what |
|---|---|---|
| 1 | `train_mnist.py` | trains the ANN and the SNN for each seed; writes `runs/manifest.json` |
| 2 | `benchmark.py` | PC benchmark: accuracy, parameters, MACs / synaptic operations, spikes |
| 3 | `pynq/export_model.py` | int8 export and the bit-exact PC integer reference |
| 4 | `pynq/benchmark_pynq.py` | runs every model on the board at a chosen firmware level |

Board tooling, firmware levels and quantisation are described in
[`pynq/README.md`](pynq/README.md).

## Reproduce

Git Bash, from this folder:

```bash
./setup_venv.sh && source .venv/Scripts/activate   # .venv/bin/activate on Linux
python train_mnist.py --seeds 0 1                  # 5 epochs, hidden 128, T = 25 (defaults)
python benchmark.py                                # PC benchmark -> runs/summary.md

# on the board (PYNQ-Z2 reachable over SSH, see pynq/README.md)
pynq/build_fw.sh                                   # only after firmware changes
python pynq/benchmark_pynq.py                      # optimised firmware, full test set
python pynq/benchmark_pynq.py --fw baseline --n 1000
python pynq/benchmark_pynq.py --fw mac             # both coprocessors
```

## Live demo

`demo/start_demo.bat` (double-click) or `demo/start_demo.sh [--offline]` opens
http://localhost:8000. You draw a digit and the ANN and the SNN classify it on the
PYNQ-Z2. The page shows board latency, cycle counts and the spike raster, and has a
switch between software and custom instructions for the SNN. `--offline` runs the PC
int8 models only.
