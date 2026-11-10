PicoRV32 RV32IM @ 100 MHz on the PYNQ-Z2, firmware fast, 10000 MNIST test images per seed, seeds [0, 1]

| metric (mean +- std over seeds) | SNN on board |
|---|---|
| board test accuracy [%] | 97.65 +- 0.11 |
| board test loss | 0.0772 +- 0.0029 |
| PC float accuracy, same samples [%] | 97.64 +- 0.07 |
| board == PC int8 reference [%] | 100.00 +- 0.00 |
| parameters | 101770 +- 0 |
| SOPs/sample | 336,608 +- 402 |
| hidden spikes/sample | 412.3 +- 42.7 |
| output spikes/sample | 22.9 +- 0.6 |
| board latency [ms/sample] | 57.13 +- 0.10 |
| board cycles/sample | 5,712,887 +- 10,229 |
| model size in BRAM [B] | 103024 +- 0 |

SOPs as in benchmark.py (one add per spike per fan-out synapse, over all timesteps). Loss = cross-entropy on the output spike counts.
