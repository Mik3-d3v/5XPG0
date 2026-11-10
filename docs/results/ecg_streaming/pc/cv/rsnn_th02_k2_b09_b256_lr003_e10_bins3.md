CV (5 patient folds on DS1, pooled out-of-fold), epochs=10, seeds=[0, 1, 2], batch=256, lr=0.003, wd=0.2, hidden=64, timesteps=25, beta=0.9, encoding=direct, use_rr=True, morph=diff, augment=True

| metric (mean +- std over seeds) | RSNN |
|---|---|
| macro_f1 | 0.627 +- 0.004 |
| accuracy | 0.910 +- 0.002 |
| se_N | 0.933 +- 0.003 |
| se_S | 0.454 +- 0.031 |
| se_V | 0.749 +- 0.010 |
| ppv_N | 0.985 +- 0.000 |
| ppv_S | 0.167 +- 0.011 |
| ppv_V | 0.620 +- 0.003 |

| record | RSNN acc |
|---|---|
| 101 | 0.998 |
| 106 | 0.919 |
| 108 | 0.935 |
| 109 (BBB) | 0.927 |
| 112 | 0.998 |
| 114 | 0.987 |
| 115 | 0.998 |
| 116 | 0.995 |
| 118 (BBB) | 0.965 |
| 119 | 0.991 |
| 122 | 0.961 |
| 124 (BBB) | 0.953 |
| 201 | 0.802 |
| 203 | 0.658 |
| 205 | 0.987 |
| 207 (BBB) | 0.851 |
| 208 | 0.760 |
| 209 | 0.928 |
| 215 | 0.869 |
| 220 | 0.989 |
| 223 | 0.925 |
| 230 | 0.753 |
