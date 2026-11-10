CV (5 patient folds on DS1, pooled out-of-fold), epochs=10, seeds=[0, 1, 2], batch=64, lr=0.001, wd=0.2, hidden=64, timesteps=25, beta=0.9, encoding=direct, use_rr=True, morph=diff, augment=False

| metric (mean +- std over seeds) | ANN |
|---|---|
| macro_f1 | 0.661 +- 0.005 |
| accuracy | 0.942 +- 0.001 |
| se_N | 0.968 +- 0.001 |
| se_S | 0.294 +- 0.009 |
| se_V | 0.782 +- 0.007 |
| ppv_N | 0.980 +- 0.000 |
| ppv_S | 0.180 +- 0.009 |
| ppv_V | 0.791 +- 0.004 |

| record | ANN acc |
|---|---|
| 101 | 0.999 |
| 106 | 0.900 |
| 108 | 0.913 |
| 109 (BBB) | 0.995 |
| 112 | 0.999 |
| 114 | 0.996 |
| 115 | 1.000 |
| 116 | 0.994 |
| 118 (BBB) | 0.975 |
| 119 | 1.000 |
| 122 | 1.000 |
| 124 (BBB) | 0.972 |
| 201 | 0.881 |
| 203 | 0.804 |
| 205 | 0.986 |
| 207 (BBB) | 0.842 |
| 208 | 0.917 |
| 209 | 0.909 |
| 215 | 0.918 |
| 220 | 0.971 |
| 223 | 0.924 |
| 230 | 0.871 |
