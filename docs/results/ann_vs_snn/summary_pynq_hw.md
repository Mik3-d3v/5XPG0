PicoRV32 RV32IM @ 100 MHz on the PYNQ-Z2, firmware hw, 10000 MNIST test images per seed, seeds [0, 1]

| metric (mean +- std over seeds) | SNN |
|---|---|
| board test accuracy [%] | 97.65 +- 0.11 |
| board test loss | 0.0772 +- 0.0029 |
| PC float accuracy, same images [%] | 97.64 +- 0.07 |
| board == PC int8 reference [%] | 100.00 +- 0.00 |
| parameters | 101770 +- 0 |
| ops/sample, benchmark.py (MAC or SOP) | 336,608 +- 402 |
| ops/sample executed by this firmware | 106,793 +- 376 |
| nonzero inputs / input spikes per sample | 2597.5 +- 0.2 |
| nonzero hidden / hidden spikes per sample | 412.3 +- 42.7 |
| output spikes/sample | 22.9 +- 0.6 |
| board latency [ms/sample] | 31.87 +- 0.09 |
| board latency min [ms] | 9.21 +- 0.55 |
| board latency max [ms] | 51.97 +- 0.81 |
| board cycles/sample | 3,187,379 +- 8,969 |
| cycles per executed op | 29.8 +- 0.0 |
| model size in BRAM [B] | 103024 +- 0 |

ops/sample, benchmark.py: ANN dense MACs, SNN synaptic operations (one add per spike per fan-out synapse, over all timesteps).
Executed ops: baseline firmware = the benchmark.py ops; fast firmware = ANN MACs of nonzero inputs/activations, SNN row updates of input changes (delta) + hidden spikes.
Loss: cross-entropy on the int8 logits (ANN, rescaled) / output spike counts (SNN).
