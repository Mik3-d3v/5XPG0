CV (5 patient folds on DS1, pooled out-of-fold), epochs=20, seeds=[0, 1, 2], batch=64, lr=0.001, wd=0.2, hidden=64, timesteps=25, beta=0.9, encoding=direct, use_rr=True, morph=diff, augment=False

| metric (mean +- std over seeds) | ANN |
|---|---|
| macro_f1 | 0.657 +- 0.006 |
| accuracy | 0.938 +- 0.002 |
| se_N | 0.964 +- 0.001 |
| se_S | 0.299 +- 0.009 |
| se_V | 0.783 +- 0.009 |
| ppv_N | 0.980 +- 0.001 |
| ppv_S | 0.163 +- 0.010 |
| ppv_V | 0.791 +- 0.005 |

| record | ANN acc |
|---|---|
| 101 | 0.999 |
| 106 | 0.906 |
| 108 | 0.902 |
| 109 (BBB) | 0.994 |
| 112 | 0.999 |
| 114 | 0.995 |
| 115 | 1.000 |
| 116 | 0.994 |
| 118 (BBB) | 0.971 |
| 119 | 1.000 |
| 122 | 1.000 |
| 124 (BBB) | 0.970 |
| 201 | 0.879 |
| 203 | 0.799 |
| 205 | 0.985 |
| 207 (BBB) | 0.841 |
| 208 | 0.914 |
| 209 | 0.910 |
| 215 | 0.869 |
| 220 | 0.973 |
| 223 | 0.927 |
| 230 | 0.881 |
