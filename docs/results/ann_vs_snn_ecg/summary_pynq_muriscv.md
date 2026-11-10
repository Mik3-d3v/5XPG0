PicoRV32 RV32IM @ 100 MHz on the PYNQ-Z2, firmware muriscv, 49273 MIT-BIH DS2 beats per seed, seeds [0, 1, 2], SNN T=None

| metric (mean +- std over seeds) | ANN |
|---|---|
| board macro-F1 (N/S/V) | 0.677 +- 0.004 |
| board accuracy [%] | 93.69 +- 0.10 |
| Se N [%] | 96.90 +- 0.09 |
| Se S [%] | 17.36 +- 0.55 |
| Se V [%] | 93.10 +- 0.32 |
| +P N [%] | 96.69 +- 0.03 |
| +P S [%] | 30.12 +- 1.31 |
| +P V [%] | 76.81 +- 0.89 |
| board loss (weighted CE) | 0.3045 +- 0.0063 |
| PC float macro-F1, same beats | 0.675 +- 0.004 |
| PC float accuracy, same beats [%] | 93.62 +- 0.09 |
| board == PC int8 reference [%] | 100.00 +- 0.00 |
| parameters | 8515 +- 0 |
| ops/beat, benchmark_ecg.py (MAC + SOP) | 8,448 +- 0 |
| ops/beat executed by this firmware | 8,448 +- 0 |
| nonzero int8 inputs per beat (of 129) | 103.7 +- 0.0 |
| nonzero hidden / hidden spikes per beat | 38.5 +- 1.1 |
| output spikes/beat | n/a |
| board latency [ms/beat] | 7.054 +- 0.000 |
| board latency min [ms] | 7.051 +- 0.000 |
| board latency max [ms] | 7.058 +- 0.000 |
| board cycles/beat | 705,412 +- 47 |
| cycles per executed op | 83.5 +- 0.0 |
| model size in BRAM [B] | 8820 +- 0 |

ops/beat, benchmark_ecg.py: ANN dense MACs; SNN (direct encoding) first-layer MACs once + SOPs of the hidden spikes over T steps.
Executed ops: baseline = the benchmark_ecg.py ops; fast = MACs of the nonzero int8 inputs + hidden events x fan-out.
Loss: class-weighted cross-entropy on the int8 logits (ANN, rescaled) / output spike counts (SNN), averaged per batch of 1024 beats as in benchmark_ecg.py.

ANN board confusion matrix (rows = true, cols = predicted, summed over seeds), classes ['N', 'S', 'V']:
     128541    1881    2232
       4068     956     484
        328     338    8991

Per-record board results (mean over seeds):

| record | n_N | n_S | n_V | ANN acc | ANN Se_S | ANN Se_V | ANN ms |
|---|---|---|---|---|---|---|---|
| 100 | 2237 | 33 | 1 | 0.995 | 0.69 | 1.00 | 7.05 |
| 103 | 2080 | 2 | 0 | 0.997 | 0.50 | - | 7.05 |
| 105 | 2524 | 0 | 41 | 0.931 | - | 0.27 | 7.05 |
| 111 | 2121 | 0 | 1 | 0.959 | - | 1.00 | 7.05 |
| 113 | 1787 | 6 | 0 | 1.000 | 1.00 | - | 7.05 |
| 117 | 1532 | 1 | 0 | 1.000 | 1.00 | - | 7.05 |
| 121 | 1859 | 1 | 1 | 0.994 | 0.67 | 1.00 | 7.05 |
| 123 | 1513 | 0 | 3 | 0.998 | - | 1.00 | 7.05 |
| 200 | 1742 | 30 | 825 | 0.976 | 0.22 | 0.97 | 7.05 |
| 202 | 2059 | 55 | 19 | 0.920 | 0.79 | 1.00 | 7.05 |
| 210 | 2421 | 22 | 195 | 0.976 | 0.24 | 0.92 | 7.05 |
| 212 | 2746 | 0 | 0 | 0.873 | - | - | 7.05 |
| 213 | 2639 | 28 | 220 | 0.975 | 0.01 | 0.82 | 7.05 |
| 214 | 2001 | 0 | 256 | 0.997 | - | 0.98 | 7.05 |
| 219 | 2080 | 7 | 64 | 0.972 | 0.29 | 0.81 | 7.05 |
| 221 | 2029 | 0 | 396 | 0.996 | - | 0.99 | 7.05 |
| 222 | 2272 | 209 | 0 | 0.808 | 0.71 | - | 7.05 |
| 228 | 1686 | 3 | 362 | 0.991 | 0.78 | 1.00 | 7.05 |
| 231 | 1566 | 1 | 2 | 0.960 | 0.00 | 1.00 | 7.05 |
| 232 | 397 | 1381 | 0 | 0.262 | 0.05 | - | 7.05 |
| 233 | 2229 | 7 | 830 | 0.966 | 0.62 | 0.89 | 7.05 |
| 234 | 2698 | 50 | 3 | 0.983 | 0.10 | 1.00 | 7.05 |

Paired per-beat comparison (same DS2 beat, same seed):

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| ANN / ANN latency, mean of totals | 1.00 | 1.00 | 1.00 |
| ANN / ANN latency, median per beat | 1.00 | 1.00 | 1.00 |
| min / max per beat | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| beats where the ANN is faster | 0 | 0 | 0 |
| corr(ANN, ANN latency) | 1.000 | 1.000 | 1.000 |
| both correct | 46,109 | 46,202 | 46,177 |
| only ANN correct | 0 | 0 | 0 | 0 | 0 | 0 |
| both wrong | 3,164 | 3,071 | 3,096 |
| same prediction | 49,273 | 49,273 | 49,273 |
