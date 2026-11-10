PicoRV32 RV32IM @ 100 MHz on the PYNQ-Z2, firmware fast, 10000 MNIST test images per seed, seeds [0, 1]

| metric (mean +- std over seeds) | ANN | SNN |
|---|---|---|
| board test accuracy [%] | 96.84 +- 0.16 | 97.65 +- 0.11 |
| board test loss | 0.1090 +- 0.0031 | 0.0772 +- 0.0029 |
| PC float accuracy, same images [%] | 96.82 +- 0.18 | 97.64 +- 0.07 |
| board == PC int8 reference [%] | 100.00 +- 0.00 | 100.00 +- 0.00 |
| parameters | 101770 +- 0 | 101770 +- 0 |
| ops/sample, benchmark.py (MAC or SOP) | 101,632 +- 0 | 336,608 +- 402 |
| ops/sample executed by this firmware | 20,167 +- 1 | 106,793 +- 376 |
| nonzero inputs / input spikes per sample | 151.1 +- 0.0 | 2597.5 +- 0.2 |
| nonzero hidden / hidden spikes per sample | 82.4 +- 0.1 | 412.3 +- 42.7 |
| output spikes/sample | n/a | 22.9 +- 0.6 |
| board latency [ms/sample] | 13.62 +- 0.00 | 57.13 +- 0.10 |
| board latency min [ms] | 5.51 +- 0.07 | 23.37 +- 0.58 |
| board latency max [ms] | 25.59 +- 0.03 | 79.11 +- 0.86 |
| board cycles/sample | 1,362,208 +- 11 | 5,712,866 +- 10,229 |
| cycles per executed op | 67.5 +- 0.0 | 53.5 +- 0.1 |
| model size in BRAM [B] | 103024 +- 0 | 103024 +- 0 |

ops/sample, benchmark.py: ANN dense MACs, SNN synaptic operations (one add per spike per fan-out synapse, over all timesteps).
Executed ops: baseline firmware = the benchmark.py ops; fast firmware = ANN MACs of nonzero inputs/activations, SNN row updates of input changes (delta) + hidden spikes.
Loss: cross-entropy on the int8 logits (ANN, rescaled) / output spike counts (SNN).
