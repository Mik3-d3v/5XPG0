CV (5 patient folds on DS1, pooled out-of-fold), epochs=5, seeds=[0, 1, 2], batch=64, lr=0.001, wd=0.2, hidden=64, timesteps=25, beta=0.9, encoding=direct, use_rr=True, morph=diff, augment=True

| metric (mean +- std over seeds) | ANN |
|---|---|
| macro_f1 | 0.664 +- 0.001 |
| accuracy | 0.941 +- 0.000 |
| se_N | 0.965 +- 0.001 |
| se_S | 0.296 +- 0.006 |
| se_V | 0.818 +- 0.006 |
| ppv_N | 0.983 +- 0.000 |
| ppv_S | 0.218 +- 0.002 |
| ppv_V | 0.720 +- 0.004 |

| record | ANN acc |
|---|---|
| 101 | 0.999 |
| 106 | 0.885 |
| 108 | 0.852 |
| 109 (BBB) | 0.996 |
| 112 | 0.999 |
| 114 | 0.995 |
| 115 | 1.000 |
| 116 | 0.995 |
| 118 (BBB) | 0.978 |
| 119 | 1.000 |
| 122 | 1.000 |
| 124 (BBB) | 0.969 |
| 201 | 0.857 |
| 203 | 0.776 |
| 205 | 0.995 |
| 207 (BBB) | 0.853 |
| 208 | 0.924 |
| 209 | 0.902 |
| 215 | 0.991 |
| 220 | 0.967 |
| 223 | 0.940 |
| 230 | 0.841 |
