CV (5 patient folds on DS1, pooled out-of-fold), epochs=5, seeds=[0, 1, 2], batch=64, lr=0.001, wd=0.05, hidden=64, timesteps=25, beta=0.9, encoding=direct, use_rr=True, morph=both, augment=False

| metric (mean +- std over seeds) | ANN |
|---|---|
| macro_f1 | 0.594 +- 0.009 |
| accuracy | 0.915 +- 0.009 |
| se_N | 0.938 +- 0.011 |
| se_S | 0.156 +- 0.014 |
| se_V | 0.828 +- 0.013 |
| ppv_N | 0.979 +- 0.001 |
| ppv_S | 0.107 +- 0.023 |
| ppv_V | 0.606 +- 0.062 |

| record | ANN acc |
|---|---|
| 101 | 0.998 |
| 106 | 0.887 |
| 108 | 0.649 |
| 109 (BBB) | 0.994 |
| 112 | 0.622 |
| 114 | 0.994 |
| 115 | 0.955 |
| 116 | 0.994 |
| 118 (BBB) | 0.959 |
| 119 | 1.000 |
| 122 | 1.000 |
| 124 (BBB) | 0.973 |
| 201 | 0.925 |
| 203 | 0.933 |
| 205 | 0.993 |
| 207 (BBB) | 0.581 |
| 208 | 0.968 |
| 209 | 0.889 |
| 215 | 0.882 |
| 220 | 0.969 |
| 223 | 0.945 |
| 230 | 0.971 |
