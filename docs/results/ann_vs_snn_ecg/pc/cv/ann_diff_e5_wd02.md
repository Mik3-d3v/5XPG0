CV (5 patient folds on DS1, pooled out-of-fold), epochs=5, seeds=[0, 1, 2], batch=64, lr=0.001, wd=0.2, hidden=64, timesteps=25, beta=0.9, encoding=direct, use_rr=True, morph=diff, augment=False

| metric (mean +- std over seeds) | ANN |
|---|---|
| macro_f1 | 0.663 +- 0.003 |
| accuracy | 0.945 +- 0.001 |
| se_N | 0.973 +- 0.000 |
| se_S | 0.277 +- 0.010 |
| se_V | 0.776 +- 0.009 |
| ppv_N | 0.979 +- 0.000 |
| ppv_S | 0.201 +- 0.006 |
| ppv_V | 0.786 +- 0.001 |

| record | ANN acc |
|---|---|
| 101 | 0.999 |
| 106 | 0.897 |
| 108 | 0.917 |
| 109 (BBB) | 0.995 |
| 112 | 0.999 |
| 114 | 0.996 |
| 115 | 1.000 |
| 116 | 0.995 |
| 118 (BBB) | 0.978 |
| 119 | 1.000 |
| 122 | 1.000 |
| 124 (BBB) | 0.974 |
| 201 | 0.880 |
| 203 | 0.820 |
| 205 | 0.989 |
| 207 (BBB) | 0.844 |
| 208 | 0.914 |
| 209 | 0.906 |
| 215 | 0.964 |
| 220 | 0.967 |
| 223 | 0.922 |
| 230 | 0.861 |
