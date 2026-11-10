PicoRV32 RV32IM @ 100 MHz on the PYNQ-Z2, firmware hw, 49273 MIT-BIH DS2 beats per seed, seeds [0, 1, 2], SNN T=10

| metric (mean +- std over seeds) | SNN |
|---|---|
| board macro-F1 (N/S/V) | 0.706 +- 0.007 |
| board accuracy [%] | 94.18 +- 0.11 |
| Se N [%] | 97.07 +- 0.11 |
| Se S [%] | 23.40 +- 1.81 |
| Se V [%] | 94.93 +- 0.58 |
| +P N [%] | 97.01 +- 0.06 |
| +P S [%] | 38.67 +- 2.54 |
| +P V [%] | 78.01 +- 0.71 |
| board loss (weighted CE) | 0.3053 +- 0.0107 |
| PC float macro-F1, same beats | 0.706 +- 0.006 |
| PC float accuracy, same beats [%] | 94.20 +- 0.10 |
| board == PC int8 reference [%] | 100.00 +- 0.00 |
| parameters | 8515 +- 0 |
| ops/beat, benchmark_ecg.py (MAC + SOP) | 8,450 +- 11 |
| ops/beat executed by this firmware | 6,833 +- 11 |
| nonzero int8 inputs per beat (of 129) | 103.7 +- 0.0 |
| nonzero hidden / hidden spikes per beat | 64.6 +- 3.7 |
| output spikes/beat | 5.0 +- 0.3 |
| board latency [ms/beat] | 5.316 +- 0.030 |
| board latency min [ms] | 0.851 +- 0.020 |
| board latency max [ms] | 7.040 +- 0.105 |
| board cycles/beat | 531,643 +- 2,986 |
| cycles per executed op | 77.8 +- 0.3 |
| model size in BRAM [B] | 9620 +- 0 |

ops/beat, benchmark_ecg.py: ANN dense MACs; SNN (direct encoding) first-layer MACs once + SOPs of the hidden spikes over T steps.
Executed ops: baseline = the benchmark_ecg.py ops; fast = MACs of the nonzero int8 inputs + hidden events x fan-out.
Loss: class-weighted cross-entropy on the int8 logits (ANN, rescaled) / output spike counts (SNN), averaged per batch of 1024 beats as in benchmark_ecg.py.

SNN board confusion matrix (rows = true, cols = predicted, summed over seeds), classes ['N', 'S', 'V']:
     128761    1770    2123
       3758    1289     461
        215     275    9167

Per-record board results (mean over seeds):

| record | n_N | n_S | n_V | SNN acc | SNN Se_S | SNN Se_V | SNN ms |
|---|---|---|---|---|---|---|---|
| 100 | 2237 | 33 | 1 | 0.992 | 0.45 | 1.00 | 4.77 |
| 103 | 2080 | 2 | 0 | 0.998 | 0.33 | - | 4.26 |
| 105 | 2524 | 0 | 41 | 0.944 | - | 0.29 | 5.10 |
| 111 | 2121 | 0 | 1 | 0.963 | - | 1.00 | 5.78 |
| 113 | 1787 | 6 | 0 | 0.997 | 0.11 | - | 4.72 |
| 117 | 1532 | 1 | 0 | 1.000 | 1.00 | - | 5.19 |
| 121 | 1859 | 1 | 1 | 0.994 | 1.00 | 1.00 | 5.19 |
| 123 | 1513 | 0 | 3 | 0.998 | - | 1.00 | 4.78 |
| 200 | 1742 | 30 | 825 | 0.976 | 0.16 | 0.97 | 5.93 |
| 202 | 2059 | 55 | 19 | 0.932 | 0.75 | 0.95 | 5.22 |
| 210 | 2421 | 22 | 195 | 0.980 | 0.14 | 0.92 | 5.37 |
| 212 | 2746 | 0 | 0 | 0.871 | - | - | 5.57 |
| 213 | 2639 | 28 | 220 | 0.982 | 0.04 | 0.91 | 5.58 |
| 214 | 2001 | 0 | 256 | 0.994 | - | 0.95 | 4.88 |
| 219 | 2080 | 7 | 64 | 0.968 | 0.24 | 0.81 | 5.09 |
| 221 | 2029 | 0 | 396 | 0.995 | - | 0.99 | 5.44 |
| 222 | 2272 | 209 | 0 | 0.807 | 0.72 | - | 6.28 |
| 228 | 1686 | 3 | 362 | 0.992 | 0.78 | 1.00 | 6.10 |
| 231 | 1566 | 1 | 2 | 0.969 | 0.00 | 1.00 | 5.01 |
| 232 | 397 | 1381 | 0 | 0.334 | 0.14 | - | 5.68 |
| 233 | 2229 | 7 | 830 | 0.978 | 0.57 | 0.94 | 5.80 |
| 234 | 2698 | 50 | 3 | 0.984 | 0.14 | 1.00 | 4.54 |
