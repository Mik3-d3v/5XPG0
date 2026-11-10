PicoRV32 RV32IM @ 100 MHz on the PYNQ-Z2, firmware baseline, 1000 MNIST test images per seed, seeds [0, 1]

| metric (mean +- std over seeds) | ANN | SNN |
|---|---|---|
| board test accuracy [%] | 96.95 +- 0.64 | 98.00 +- 0.42 |
| board test loss | 0.1031 +- 0.0048 | 0.0657 +- 0.0028 |
| PC float accuracy, same images [%] | 96.85 +- 0.78 | 97.95 +- 0.21 |
| board == PC int8 reference [%] | 100.00 +- 0.00 | 100.00 +- 0.00 |
| parameters | 101770 +- 0 | 101770 +- 0 |
| ops/sample, benchmark.py (MAC or SOP) | 101,632 +- 0 | 310,720 +- 475 |
| ops/sample executed by this firmware | 101,632 +- 0 | 310,720 +- 475 |
| nonzero inputs / input spikes per sample | 142.4 +- 0.0 | 2396.3 +- 0.3 |
| nonzero hidden / hidden spikes per sample | 82.6 +- 0.1 | 399.3 +- 43.5 |
| output spikes/sample | n/a | 22.6 +- 0.5 |
| board latency [ms/sample] | 135.79 +- 0.00 | 402.65 +- 0.62 |
| board latency min [ms] | 135.74 +- 0.00 | 130.18 +- 3.15 |
| board latency max [ms] | 135.84 +- 0.00 | 893.43 +- 1.33 |
| board cycles/sample | 13,579,483 +- 9 | 40,264,672 +- 61,542 |
| cycles per executed op | 133.6 +- 0.0 | 129.6 +- 0.0 |
| model size in BRAM [B] | 103024 +- 0 | 103024 +- 0 |

ops/sample, benchmark.py: ANN dense MACs, SNN synaptic operations (one add per spike per fan-out synapse, over all timesteps).
Executed ops: baseline firmware = the benchmark.py ops; fast firmware = ANN MACs of nonzero inputs/activations, SNN row updates of input changes (delta) + hidden spikes.
Loss: cross-entropy on the int8 logits (ANN, rescaled) / output spike counts (SNN).
