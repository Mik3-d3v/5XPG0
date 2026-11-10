epochs=5, seeds=[0, 1], batch=64, lr=0.001, hidden=128, timesteps=25, beta=0.9

| metric (mean +- std over seeds) | ANN | SNN |
|---|---|---|
| test accuracy [%] | 96.82 +- 0.18 | 97.64 +- 0.07 |
| test loss | 0.1091 +- 0.0033 | 0.0776 +- 0.0026 |
| val accuracy [%] | 96.26 +- 0.06 | 97.30 +- 0.06 |
| parameters | 101770 +- 0 | 101770 +- 0 |
| ops/sample (MAC or SOP) | 101,632 +- 0 | 336,614 +- 407 |
| hidden spikes/sample | n/a | 412.9 +- 43.1 |
| output spikes/sample | n/a | 22.9 +- 0.7 |
| PC latency [ms/sample] | 0.0014 +- 0.0001 | 0.2655 +- 0.0156 |
| train time [s] | 47.8 +- 15.1 | 237.5 +- 3.5 |

ANN ops are dense MACs; SNN ops are synaptic operations (one add per spike per fan-out synapse, over all timesteps).
