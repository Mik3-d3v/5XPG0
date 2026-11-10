PicoRV32 RV32IM @ 100 MHz on the PYNQ-Z2, firmware muriscv, 10000 MNIST test images per seed, seeds [0, 1]

| metric (mean +- std over seeds) | ANN |
|---|---|
| board test accuracy [%] | 96.85 +- 0.10 |
| board test loss | 0.1090 +- 0.0029 |
| PC float accuracy, same images [%] | 96.82 +- 0.18 |
| board == PC int8 reference [%] | 100.00 +- 0.00 |
| parameters | 101770 +- 0 |
| ops/sample, benchmark.py (MAC or SOP) | 101,632 +- 0 |
| ops/sample executed by this firmware | 101,632 +- 0 |
| nonzero inputs / input spikes per sample | 151.1 +- 0.0 |
| nonzero hidden / hidden spikes per sample | 82.4 +- 0.1 |
| output spikes/sample | n/a |
| board latency [ms/sample] | 76.62 +- 0.00 |
| board latency min [ms] | 76.61 +- 0.00 |
| board latency max [ms] | 76.62 +- 0.00 |
| board cycles/sample | 7,661,847 +- 20 |
| cycles per executed op | 75.4 +- 0.0 |
| model size in BRAM [B] | 102288 +- 0 |

ops/sample, benchmark.py: ANN dense MACs, SNN synaptic operations (one add per spike per fan-out synapse, over all timesteps).
Executed ops: baseline firmware = the benchmark.py ops; fast firmware = ANN MACs of nonzero inputs/activations, SNN row updates of input changes (delta) + hidden spikes.
Loss: cross-entropy on the int8 logits (ANN, rescaled) / output spike counts (SNN).
