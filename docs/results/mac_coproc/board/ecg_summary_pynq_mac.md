PicoRV32 RV32IM @ 100 MHz on the PYNQ-Z2, firmware mac, 49273 MIT-BIH DS2 beats per seed, seeds [0, 1, 2], SNN T=10

| metric (mean +- std over seeds) | ANN | SNN |
|---|---|---|
| board macro-F1 (N/S/V) | 0.675 +- 0.004 | 0.706 +- 0.007 |
| board accuracy [%] | 93.62 +- 0.12 | 94.18 +- 0.11 |
| Se N [%] | 96.80 +- 0.11 | 97.07 +- 0.11 |
| Se S [%] | 17.30 +- 0.46 | 23.40 +- 1.81 |
| Se V [%] | 93.34 +- 0.20 | 94.93 +- 0.58 |
| +P N [%] | 96.73 +- 0.04 | 97.01 +- 0.06 |
| +P S [%] | 29.72 +- 1.51 | 38.67 +- 2.54 |
| +P V [%] | 76.08 +- 1.01 | 78.01 +- 0.71 |
| board loss (weighted CE) | 0.3044 +- 0.0063 | 0.3053 +- 0.0107 |
| PC float macro-F1, same beats | 0.675 +- 0.004 | 0.706 +- 0.006 |
| PC float accuracy, same beats [%] | 93.62 +- 0.09 | 94.20 +- 0.10 |
| board == PC int8 reference [%] | 100.00 +- 0.00 | 100.00 +- 0.00 |
| parameters | 8515 +- 0 | 8515 +- 0 |
| ops/beat, benchmark_ecg.py (MAC + SOP) | 8,448 +- 0 | 8,450 +- 11 |
| ops/beat executed by this firmware | 6,754 +- 3 | 6,833 +- 11 |
| nonzero int8 inputs per beat (of 129) | 103.7 +- 0.0 | 103.7 +- 0.0 |
| nonzero hidden / hidden spikes per beat | 38.5 +- 1.1 | 64.6 +- 3.7 |
| output spikes/beat | n/a | 5.0 +- 0.3 |
| board latency [ms/beat] | 2.134 +- 0.007 | 2.658 +- 0.030 |
| board latency min [ms] | 0.625 +- 0.019 | 0.740 +- 0.020 |
| board latency max [ms] | 2.562 +- 0.016 | 3.814 +- 0.114 |
| board cycles/beat | 213,436 +- 660 | 265,800 +- 2,986 |
| cycles per executed op | 31.6 +- 0.1 | 38.9 +- 0.4 |
| model size in BRAM [B] | 9620 +- 0 | 9620 +- 0 |

ops/beat, benchmark_ecg.py: ANN dense MACs; SNN (direct encoding) first-layer MACs once + SOPs of the hidden spikes over T steps.
Executed ops: baseline = the benchmark_ecg.py ops; fast = MACs of the nonzero int8 inputs + hidden events x fan-out.
Loss: class-weighted cross-entropy on the int8 logits (ANN, rescaled) / output spike counts (SNN), averaged per batch of 1024 beats as in benchmark_ecg.py.

ANN board confusion matrix (rows = true, cols = predicted, summed over seeds), classes ['N', 'S', 'V']:
     128414    1928    2312
       4032     953     523
        315     328    9014

SNN board confusion matrix (rows = true, cols = predicted, summed over seeds), classes ['N', 'S', 'V']:
     128761    1770    2123
       3758    1289     461
        215     275    9167

Per-record board results (mean over seeds):

| record | n_N | n_S | n_V | ANN acc | ANN Se_S | ANN Se_V | ANN ms | SNN acc | SNN Se_S | SNN Se_V | SNN ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 2237 | 33 | 1 | 0.995 | 0.72 | 1.00 | 1.97 | 0.992 | 0.45 | 1.00 | 2.38 |
| 103 | 2080 | 2 | 0 | 0.997 | 0.50 | - | 1.78 | 0.998 | 0.33 | - | 2.18 |
| 105 | 2524 | 0 | 41 | 0.930 | - | 0.29 | 2.09 | 0.944 | - | 0.29 | 2.54 |
| 111 | 2121 | 0 | 1 | 0.956 | - | 1.00 | 2.26 | 0.963 | - | 1.00 | 2.90 |
| 113 | 1787 | 6 | 0 | 1.000 | 1.00 | - | 1.92 | 0.997 | 0.11 | - | 2.41 |
| 117 | 1532 | 1 | 0 | 1.000 | 1.00 | - | 2.07 | 1.000 | 1.00 | - | 2.61 |
| 121 | 1859 | 1 | 1 | 0.994 | 0.67 | 1.00 | 2.14 | 0.994 | 1.00 | 1.00 | 2.53 |
| 123 | 1513 | 0 | 3 | 0.998 | - | 1.00 | 1.95 | 0.998 | - | 1.00 | 2.41 |
| 200 | 1742 | 30 | 825 | 0.975 | 0.23 | 0.97 | 2.30 | 0.976 | 0.16 | 0.97 | 2.98 |
| 202 | 2059 | 55 | 19 | 0.917 | 0.76 | 1.00 | 2.11 | 0.932 | 0.75 | 0.95 | 2.61 |
| 210 | 2421 | 22 | 195 | 0.975 | 0.21 | 0.92 | 2.20 | 0.980 | 0.14 | 0.92 | 2.63 |
| 212 | 2746 | 0 | 0 | 0.872 | - | - | 2.23 | 0.871 | - | - | 2.76 |
| 213 | 2639 | 28 | 220 | 0.975 | 0.01 | 0.83 | 2.23 | 0.982 | 0.04 | 0.91 | 2.79 |
| 214 | 2001 | 0 | 256 | 0.997 | - | 0.98 | 2.01 | 0.994 | - | 0.95 | 2.45 |
| 219 | 2080 | 7 | 64 | 0.970 | 0.29 | 0.81 | 2.09 | 0.968 | 0.24 | 0.81 | 2.52 |
| 221 | 2029 | 0 | 396 | 0.996 | - | 0.99 | 2.17 | 0.995 | - | 0.99 | 2.72 |
| 222 | 2272 | 209 | 0 | 0.804 | 0.71 | - | 2.37 | 0.807 | 0.72 | - | 3.19 |
| 228 | 1686 | 3 | 362 | 0.991 | 1.00 | 1.00 | 2.35 | 0.992 | 0.78 | 1.00 | 3.05 |
| 231 | 1566 | 1 | 2 | 0.958 | 0.00 | 1.00 | 2.03 | 0.969 | 0.00 | 1.00 | 2.51 |
| 232 | 397 | 1381 | 0 | 0.262 | 0.05 | - | 2.25 | 0.334 | 0.14 | - | 2.83 |
| 233 | 2229 | 7 | 830 | 0.967 | 0.62 | 0.89 | 2.27 | 0.978 | 0.57 | 0.94 | 2.91 |
| 234 | 2698 | 50 | 3 | 0.983 | 0.09 | 1.00 | 1.94 | 0.984 | 0.14 | 1.00 | 2.26 |

Paired per-beat comparison (same DS2 beat, same seed):

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| SNN / ANN latency, mean of totals | 1.26 | 1.24 | 1.24 |
| SNN / ANN latency, median per beat | 1.25 | 1.23 | 1.23 |
| min / max per beat | 0.98 / 1.69 | 0.97 / 1.63 | 1.00 / 1.60 |
| beats where the SNN is faster | 1 | 5 | 0 |
| corr(ANN, SNN latency) | 0.875 | 0.889 | 0.893 |
| both correct | 45,744 | 45,853 | 45,858 |
| only ANN correct | 316 | 321 | 289 |
| only SNN correct | 605 | 556 | 601 |
| both wrong | 2,608 | 2,543 | 2,525 |
| same prediction | 48,225 | 48,264 | 48,256 |
