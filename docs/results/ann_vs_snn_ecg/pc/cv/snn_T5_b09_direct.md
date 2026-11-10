CV (5 patient folds on DS1, pooled out-of-fold), epochs=10, seeds=[0, 1, 2], batch=64, lr=0.001, wd=0.2, hidden=64, timesteps=5, beta=0.9, encoding=direct, use_rr=True, morph=diff, augment=True

| metric (mean +- std over seeds) | SNN |
|---|---|
| macro_f1 | 0.652 +- 0.004 |
| accuracy | 0.939 +- 0.000 |
| se_N | 0.965 +- 0.002 |
| se_S | 0.279 +- 0.012 |
| se_V | 0.784 +- 0.015 |
| ppv_N | 0.981 +- 0.001 |
| ppv_S | 0.205 +- 0.005 |
| ppv_V | 0.711 +- 0.001 |

| record | SNN acc |
|---|---|
| 101 | 0.998 |
| 106 | 0.882 |
| 108 | 0.861 |
| 109 (BBB) | 0.995 |
| 112 | 0.999 |
| 114 | 0.994 |
| 115 | 1.000 |
| 116 | 0.996 |
| 118 (BBB) | 0.975 |
| 119 | 1.000 |
| 122 | 1.000 |
| 124 (BBB) | 0.972 |
| 201 | 0.875 |
| 203 | 0.763 |
| 205 | 0.994 |
| 207 (BBB) | 0.845 |
| 208 | 0.914 |
| 209 | 0.906 |
| 215 | 0.975 |
| 220 | 0.972 |
| 223 | 0.926 |
| 230 | 0.834 |
