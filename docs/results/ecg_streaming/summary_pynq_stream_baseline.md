PicoRV32 RV32IM @ 100 MHz on the PYNQ-Z2, firmware baseline, 200 MIT-BIH DS2 beats per seed, seeds [0, 1, 2], SNN T=None

| metric (mean +- std over seeds) | GRU | RSNN |
|---|---|---|
| board macro-F1 (N/S/V) | 0.329 +- 0.001 | 0.329 +- 0.000 |
| board accuracy [%] | 97.33 +- 0.76 | 97.50 +- 0.00 |
| Se N [%] | 97.33 +- 0.76 | 97.50 +- 0.00 |
| Se S [%] | 0.00 +- 0.00 | 0.00 +- 0.00 |
| Se V [%] | 0.00 +- 0.00 | 0.00 +- 0.00 |
| +P N [%] | 100.00 +- 0.00 | 100.00 +- 0.00 |
| +P S [%] | 0.00 +- 0.00 | 0.00 +- 0.00 |
| +P V [%] | 0.00 +- 0.00 | 0.00 +- 0.00 |
| board loss (weighted CE) | 0.1151 +- 0.0143 | 0.2107 +- 0.0057 |
| PC float macro-F1, same beats | 0.329 +- 0.002 | 0.330 +- 0.001 |
| PC float accuracy, same beats [%] | 97.67 +- 1.04 | 98.00 +- 0.50 |
| board == PC int8 reference [%] | 100.00 +- 0.00 | 100.00 +- 0.00 |
| parameters | 13452 +- 0 | 4940 +- 0 |
| ops/beat, benchmark_ecg.py (MAC + SOP) | 1,573,065 +- 0 | 1,032 +- 62 |
| ops/beat executed by this firmware | 1,573,065 +- 0 | 1,032 +- 62 |
| nonzero int8 inputs per beat (of 129) | 91.2 +- 0.0 | 6.5 +- 0.0 |
| nonzero hidden / hidden spikes per beat | 7966.5 +- 13.3 | 9.6 +- 1.0 |
| output spikes/beat | n/a | n/a |
| board latency [ms/beat] | 2203.089 +- 0.000 | 35.210 +- 0.078 |
| board latency min [ms] | 2203.089 +- 0.000 | 33.917 +- 0.000 |
| board latency max [ms] | 2203.090 +- 0.000 | 43.935 +- 0.001 |
| board cycles/beat | 220,308,932 +- 2 | 3,521,005 +- 7,785 |
| cycles per executed op | 140.1 +- 0.0 | 3420.5 +- 199.2 |
| model size in BRAM [B] | 19336 +- 0 | 7852 +- 0 |

ops/beat, benchmark_ecg.py: ANN dense MACs; SNN (direct encoding) first-layer MACs once + SOPs of the hidden spikes over T steps.
Executed ops: baseline = the benchmark_ecg.py ops; fast = MACs of the nonzero int8 inputs + hidden events x fan-out.
Loss: class-weighted cross-entropy on the int8 logits (ANN, rescaled) / output spike counts (SNN), averaged per batch of 1024 beats as in benchmark_ecg.py.

GRU board confusion matrix (rows = true, cols = predicted, summed over seeds), classes ['N', 'S', 'V']:
        584       1      15
          0       0       0
          0       0       0

RSNN board confusion matrix (rows = true, cols = predicted, summed over seeds), classes ['N', 'S', 'V']:
        585       0      15
          0       0       0
          0       0       0

Per-record board results (mean over seeds):

| record | n_N | n_S | n_V | GRU acc | GRU Se_S | GRU Se_V | GRU ms | RSNN acc | RSNN Se_S | RSNN Se_V | RSNN ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 39 | 0 | 0 | 1.000 | - | - | 2203.09 | 1.000 | - | - | 35.61 |
| 103 | 47 | 0 | 0 | 1.000 | - | - | 2203.09 | 1.000 | - | - | 35.03 |
| 105 | 46 | 0 | 0 | 0.899 | - | - | 2203.09 | 0.906 | - | - | 35.13 |
| 111 | 37 | 0 | 0 | 0.982 | - | - | 2203.09 | 1.000 | - | - | 35.38 |
| 113 | 31 | 0 | 0 | 1.000 | - | - | 2203.09 | 0.978 | - | - | 34.89 |

Paired per-beat comparison (same DS2 beat, same seed):

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| RSNN / GRU latency, mean of totals | 0.02 | 0.02 | 0.02 |
| RSNN / GRU latency, median per beat | 0.02 | 0.02 | 0.02 |
| min / max per beat | 0.02 / 0.02 | 0.02 / 0.02 | 0.02 / 0.02 |
| beats where the RSNN is faster | 200 | 200 | 200 |
| corr(GRU, RSNN latency) | 0.353 | 0.400 | 0.313 |
| both correct | 193 | 190 | 193 |
| only GRU correct | 3 | 3 | 2 |
| only RSNN correct | 2 | 5 | 2 |
| both wrong | 2 | 2 | 3 |
| same prediction | 195 | 191 | 196 |
