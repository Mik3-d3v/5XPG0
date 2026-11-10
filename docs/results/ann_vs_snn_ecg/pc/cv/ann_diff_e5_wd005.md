CV (5 patient folds on DS1, pooled out-of-fold), epochs=5, seeds=[0, 1, 2], batch=64, lr=0.001, wd=0.05, hidden=64, timesteps=25, beta=0.9, encoding=direct, use_rr=True, morph=diff, augment=False

| metric (mean +- std over seeds) | ANN |
|---|---|
| macro_f1 | 0.662 +- 0.003 |
| accuracy | 0.943 +- 0.001 |
| se_N | 0.970 +- 0.000 |
| se_S | 0.288 +- 0.003 |
| se_V | 0.777 +- 0.009 |
| ppv_N | 0.980 +- 0.000 |
| ppv_S | 0.190 +- 0.006 |
| ppv_V | 0.787 +- 0.001 |

| record | ANN acc |
|---|---|
| 101 | 0.999 |
| 106 | 0.894 |
| 108 | 0.926 |
| 109 (BBB) | 0.995 |
| 112 | 0.999 |
| 114 | 0.996 |
| 115 | 1.000 |
| 116 | 0.995 |
| 118 (BBB) | 0.981 |
| 119 | 1.000 |
| 122 | 1.000 |
| 124 (BBB) | 0.973 |
| 201 | 0.875 |
| 203 | 0.807 |
| 205 | 0.987 |
| 207 (BBB) | 0.841 |
| 208 | 0.917 |
| 209 | 0.906 |
| 215 | 0.946 |
| 220 | 0.968 |
| 223 | 0.923 |
| 230 | 0.859 |
