# overnight job C (full)

| job | workload | model | hidden | T | fw | seeds | params | model B | metric | PC float | PC int8 | CV macro-F1 | board n/seed | board ms (p5–p95) | bit-exact | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C | ecg-stream | GRU | 16 |  | fast | 3 | 1068 | 4216 | macro-F1 (all DS2) | 0.6726 ± 0.0099 | 0.6753 ± 0.0082 |  | 200/200/200 | 112.222 (111.911–112.435) | 600/600 | ecgC_gru_h16 |
| C | ecg-stream | GRU | 32 |  | fast | 3 | 3660 | 7720 | macro-F1 (all DS2) | 0.6598 ± 0.0091 | 0.6608 ± 0.0076 |  | 200/200/200 | 331.801 (330.834–332.664) | 600/600 | ecgC_gru_h32 |
