CV (5 patient folds on DS1, pooled out-of-fold), epochs=10, seeds=[0, 1, 2], batch=64, lr=0.001, wd=0.0, hidden=64, timesteps=25, beta=0.9, encoding=direct, use_rr=True, morph=diff, augment=False

| metric (mean +- std over seeds) | ANN |
|---|---|
| macro_f1 | 0.653 +- 0.006 |
| accuracy | 0.935 +- 0.002 |
| se_N | 0.961 +- 0.001 |
| se_S | 0.305 +- 0.007 |
| se_V | 0.780 +- 0.014 |
| ppv_N | 0.980 +- 0.000 |
| ppv_S | 0.154 +- 0.008 |
| ppv_V | 0.785 +- 0.006 |

| record | ANN acc |
|---|---|
| 101 | 0.999 |
| 106 | 0.897 |
| 108 | 0.915 |
| 109 (BBB) | 0.994 |
| 112 | 0.999 |
| 114 | 0.996 |
| 115 | 1.000 |
| 116 | 0.994 |
| 118 (BBB) | 0.973 |
| 119 | 1.000 |
| 122 | 1.000 |
| 124 (BBB) | 0.971 |
| 201 | 0.873 |
| 203 | 0.780 |
| 205 | 0.984 |
| 207 (BBB) | 0.841 |
| 208 | 0.920 |
| 209 | 0.909 |
| 215 | 0.851 |
| 220 | 0.972 |
| 223 | 0.927 |
| 230 | 0.865 |
