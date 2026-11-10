CV (5 patient folds on DS1, pooled out-of-fold), epochs=10, seeds=[0, 1, 2], batch=256, lr=0.003, wd=0.2, hidden=64, timesteps=25, beta=0.9, encoding=direct, use_rr=True, morph=diff, augment=True

| metric (mean +- std over seeds) | RSNN |
|---|---|
| macro_f1 | 0.665 +- 0.006 |
| accuracy | 0.939 +- 0.003 |
| se_N | 0.973 +- 0.001 |
| se_S | 0.359 +- 0.005 |
| se_V | 0.675 +- 0.018 |
| ppv_N | 0.971 +- 0.002 |
| ppv_S | 0.284 +- 0.006 |
| ppv_V | 0.741 +- 0.021 |

| record | RSNN acc |
|---|---|
| 101 | 0.998 |
| 106 | 0.914 |
| 108 | 0.896 |
| 109 (BBB) | 0.992 |
| 112 | 0.999 |
| 114 | 0.996 |
| 115 | 1.000 |
| 116 | 0.992 |
| 118 (BBB) | 0.961 |
| 119 | 0.999 |
| 122 | 1.000 |
| 124 (BBB) | 0.969 |
| 201 | 0.811 |
| 203 | 0.754 |
| 205 | 0.994 |
| 207 (BBB) | 0.841 |
| 208 | 0.797 |
| 209 | 0.932 |
| 215 | 0.990 |
| 220 | 0.972 |
| 223 | 0.885 |
| 230 | 0.994 |
