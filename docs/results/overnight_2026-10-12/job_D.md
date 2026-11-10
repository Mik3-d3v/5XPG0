# overnight job D (full)

| job | workload | model | hidden | T | fw | seeds | params | model B | metric | PC float | PC int8 | CV macro-F1 | board n/seed | board ms (p5–p95) | bit-exact | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D | mnist | ANN | 128 |  | fast | 1 | 101770 | 103024 | accuracy (10k test) | 0.9681 ± 0.0000 | 0.9676 ± 0.0000 |  | 10000 | 13.714 (8.345–18.798) | 10000/10000 | mnistD_ann |
| D | mnist | SNN | 128 | 25 | fast | 1 | 101770 | 103024 | accuracy (10k test) | 0.9757 ± 0.0000 | 0.9759 ± 0.0000 |  | 10000 | 56.840 (44.204–66.515) | 10000/10000 | mnistD_snn |
