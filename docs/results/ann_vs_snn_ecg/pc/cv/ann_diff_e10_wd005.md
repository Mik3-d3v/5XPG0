CV (5 patient folds on DS1, pooled out-of-fold), epochs=10, seeds=[0, 1, 2], batch=64, lr=0.001, wd=0.05, hidden=64, timesteps=25, beta=0.9, encoding=direct, use_rr=True, morph=diff, augment=False

| metric (mean +- std over seeds) | ANN |
|---|---|
| macro_f1 | 0.656 +- 0.004 |
| accuracy | 0.937 +- 0.001 |
| se_N | 0.963 +- 0.000 |
| se_S | 0.306 +- 0.010 |
| se_V | 0.781 +- 0.007 |
| ppv_N | 0.980 +- 0.000 |
| ppv_S | 0.161 +- 0.007 |
| ppv_V | 0.789 +- 0.003 |

| record | ANN acc |
|---|---|
| 101 | 0.999 |
| 106 | 0.897 |
| 108 | 0.918 |
| 109 (BBB) | 0.994 |
| 112 | 0.999 |
| 114 | 0.996 |
| 115 | 1.000 |
| 116 | 0.994 |
| 118 (BBB) | 0.974 |
| 119 | 1.000 |
| 122 | 1.000 |
| 124 (BBB) | 0.971 |
| 201 | 0.874 |
| 203 | 0.787 |
| 205 | 0.984 |
| 207 (BBB) | 0.841 |
| 208 | 0.921 |
| 209 | 0.908 |
| 215 | 0.869 |
| 220 | 0.973 |
| 223 | 0.926 |
| 230 | 0.869 |
