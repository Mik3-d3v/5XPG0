CV (5 patient folds on DS1, pooled out-of-fold), epochs=10, seeds=[0, 1, 2], batch=256, lr=0.003, wd=0.2, hidden=64, timesteps=25, beta=0.9, encoding=direct, use_rr=True, morph=diff, augment=True

| metric (mean +- std over seeds) | RSNN |
|---|---|
| macro_f1 | 0.679 +- 0.004 |
| accuracy | 0.940 +- 0.001 |
| se_N | 0.967 +- 0.001 |
| se_S | 0.432 +- 0.013 |
| se_V | 0.734 +- 0.008 |
| ppv_N | 0.979 +- 0.001 |
| ppv_S | 0.253 +- 0.001 |
| ppv_V | 0.753 +- 0.011 |

| record | RSNN acc |
|---|---|
| 101 | 0.998 |
| 106 | 0.892 |
| 108 | 0.960 |
| 109 (BBB) | 0.995 |
| 112 | 0.998 |
| 114 | 0.995 |
| 115 | 1.000 |
| 116 | 0.993 |
| 118 (BBB) | 0.963 |
| 119 | 1.000 |
| 122 | 1.000 |
| 124 (BBB) | 0.968 |
| 201 | 0.846 |
| 203 | 0.721 |
| 205 | 0.993 |
| 207 (BBB) | 0.857 |
| 208 | 0.818 |
| 209 | 0.952 |
| 215 | 0.970 |
| 220 | 0.977 |
| 223 | 0.913 |
| 230 | 0.924 |
