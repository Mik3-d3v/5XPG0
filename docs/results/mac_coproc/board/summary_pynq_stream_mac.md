PicoRV32 RV32IM @ 100 MHz on the PYNQ-Z2, firmware mac, 1000 MIT-BIH DS2 beats per seed, seeds [0, 1, 2], SNN T=None

| metric (mean +- std over seeds) | GRU | RSNN |
|---|---|---|
| board macro-F1 (N/S/V) | 0.664 +- 0.007 | 0.655 +- 0.009 |
| board accuracy [%] | 92.90 +- 0.56 | 93.07 +- 0.15 |
| Se N [%] | 96.04 +- 0.67 | 96.44 +- 0.19 |
| Se S [%] | 16.19 +- 1.65 | 18.10 +- 3.30 |
| Se V [%] | 90.77 +- 1.54 | 86.67 +- 1.78 |
| +P N [%] | 96.22 +- 0.19 | 96.66 +- 0.10 |
| +P S [%] | 21.67 +- 7.23 | 26.64 +- 1.52 |
| +P V [%] | 79.75 +- 2.12 | 71.91 +- 0.42 |
| board loss (weighted CE) | 0.5070 +- 0.0159 | 0.4988 +- 0.0018 |
| PC float macro-F1, same beats | 0.673 +- 0.005 | 0.666 +- 0.014 |
| PC float accuracy, same beats [%] | 93.20 +- 0.17 | 93.30 +- 0.26 |
| board == PC int8 reference [%] | 100.00 +- 0.00 | 100.00 +- 0.00 |
| parameters | 13452 +- 0 | 4940 +- 0 |
| ops/beat, benchmark_ecg.py (MAC + SOP) | 1,573,065 +- 0 | 1,530 +- 33 |
| ops/beat executed by this firmware | 1,573,065 +- 0 | 1,530 +- 33 |
| nonzero int8 inputs per beat (of 129) | 100.1 +- 0.0 | 8.5 +- 0.0 |
| nonzero hidden / hidden spikes per beat | 7970.3 +- 10.3 | 15.4 +- 0.5 |
| output spikes/beat | n/a | n/a |
| board latency [ms/beat] | 492.487 +- 0.411 | 4.582 +- 0.027 |
| board latency min [ms] | 489.633 +- 0.779 | 3.943 +- 0.000 |
| board latency max [ms] | 495.089 +- 0.165 | 10.636 +- 0.433 |
| board cycles/beat | 49,248,715 +- 41,093 | 458,154 +- 2,721 |
| cycles per executed op | 31.3 +- 0.0 | 299.4 +- 4.7 |
| model size in BRAM [B] | 19336 +- 0 | 7852 +- 0 |

ops/beat, benchmark_ecg.py: ANN dense MACs; SNN (direct encoding) first-layer MACs once + SOPs of the hidden spikes over T steps.
Executed ops: baseline = the benchmark_ecg.py ops; fast = MACs of the nonzero int8 inputs + hidden events x fan-out.
Loss: class-weighted cross-entropy on the int8 logits (ANN, rescaled) / output spike counts (SNN), averaged per batch of 1024 beats as in benchmark_ecg.py.

GRU board confusion matrix (rows = true, cols = predicted, summed over seeds), classes ['N', 'S', 'V']:
       2593      62      45
         88      17       0
         14       4     177

RSNN board confusion matrix (rows = true, cols = predicted, summed over seeds), classes ['N', 'S', 'V']:
       2604      46      50
         70      19      16
         20       6     169

Per-record board results (mean over seeds):

| record | n_N | n_S | n_V | GRU acc | GRU Se_S | GRU Se_V | GRU ms | RSNN acc | RSNN Se_S | RSNN Se_V | RSNN ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 39 | 0 | 0 | 1.000 | - | - | 492.36 | 1.000 | - | - | 4.52 |
| 103 | 47 | 0 | 0 | 1.000 | - | - | 492.23 | 1.000 | - | - | 4.33 |
| 105 | 46 | 0 | 0 | 0.899 | - | - | 492.51 | 0.906 | - | - | 4.39 |
| 111 | 37 | 0 | 0 | 0.982 | - | - | 492.35 | 1.000 | - | - | 4.50 |
| 113 | 33 | 0 | 0 | 1.000 | - | - | 492.22 | 0.980 | - | - | 4.30 |
| 117 | 22 | 0 | 0 | 1.000 | - | - | 492.58 | 1.000 | - | - | 4.65 |
| 121 | 40 | 0 | 0 | 0.975 | - | - | 492.35 | 1.000 | - | - | 4.16 |
| 123 | 38 | 0 | 0 | 1.000 | - | - | 492.44 | 1.000 | - | - | 4.29 |
| 200 | 28 | 1 | 12 | 0.951 | 0.00 | 1.00 | 492.98 | 1.000 | 1.00 | 1.00 | 5.52 |
| 202 | 47 | 1 | 0 | 0.903 | 1.00 | - | 492.11 | 0.889 | 0.00 | - | 4.18 |
| 210 | 35 | 0 | 3 | 1.000 | - | 1.00 | 492.28 | 0.947 | - | 1.00 | 4.46 |
| 212 | 60 | 0 | 0 | 0.967 | - | - | 492.35 | 1.000 | - | - | 4.33 |
| 213 | 63 | 3 | 6 | 0.944 | 0.67 | 0.72 | 492.35 | 0.949 | 0.89 | 0.44 | 4.62 |
| 214 | 40 | 0 | 7 | 1.000 | - | 1.00 | 492.29 | 0.993 | - | 0.95 | 4.42 |
| 219 | 42 | 0 | 3 | 0.993 | - | 1.00 | 492.34 | 0.926 | - | 0.56 | 4.24 |
| 221 | 43 | 0 | 8 | 0.961 | - | 0.88 | 492.44 | 0.961 | - | 0.83 | 4.57 |
| 222 | 50 | 2 | 0 | 0.718 | 1.00 | - | 493.34 | 0.673 | 1.00 | - | 5.68 |
| 228 | 36 | 0 | 8 | 0.932 | - | 1.00 | 492.97 | 0.962 | - | 0.79 | 5.36 |
| 231 | 34 | 0 | 0 | 0.971 | - | - | 492.29 | 1.000 | - | - | 4.39 |
| 232 | 9 | 26 | 0 | 0.276 | 0.03 | - | 492.62 | 0.276 | 0.03 | - | 4.47 |
| 233 | 42 | 0 | 18 | 0.944 | - | 0.81 | 492.84 | 0.972 | - | 0.96 | 5.11 |
| 234 | 69 | 2 | 0 | 0.967 | 0.00 | - | 492.38 | 0.972 | 0.00 | - | 4.17 |

Paired per-beat comparison (same DS2 beat, same seed):

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| RSNN / GRU latency, mean of totals | 0.01 | 0.01 | 0.01 |
| RSNN / GRU latency, median per beat | 0.01 | 0.01 | 0.01 |
| min / max per beat | 0.01 / 0.02 | 0.01 / 0.02 | 0.01 / 0.02 |
| beats where the RSNN is faster | 1,000 | 1,000 | 1,000 |
| corr(GRU, RSNN latency) | 0.419 | 0.520 | 0.644 |
| both correct | 908 | 909 | 905 |
| only GRU correct | 26 | 21 | 18 |
| only RSNN correct | 23 | 23 | 24 |
| both wrong | 43 | 47 | 53 |
| same prediction | 946 | 949 | 948 |
