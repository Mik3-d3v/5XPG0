CV (5 patient folds on DS1, pooled out-of-fold), epochs=10, seeds=[0, 1, 2], batch=256, lr=0.003, wd=0.2, hidden=64, timesteps=25, beta=0.9, encoding=direct, use_rr=True, morph=diff, augment=True

| metric (mean +- std over seeds) | RSNN |
|---|---|
| macro_f1 | 0.662 +- 0.005 |
| accuracy | 0.934 +- 0.001 |
| se_N | 0.962 +- 0.001 |
| se_S | 0.425 +- 0.028 |
| se_V | 0.724 +- 0.002 |
| ppv_N | 0.980 +- 0.001 |
| ppv_S | 0.227 +- 0.009 |
| ppv_V | 0.714 +- 0.004 |

| record | RSNN acc |
|---|---|
| 101 | 0.999 |
| 106 | 0.894 |
| 108 | 0.957 |
| 109 (BBB) | 0.996 |
| 112 | 0.998 |
| 114 | 0.994 |
| 115 | 1.000 |
| 116 | 0.995 |
| 118 (BBB) | 0.961 |
| 119 | 1.000 |
| 122 | 1.000 |
| 124 (BBB) | 0.969 |
| 201 | 0.844 |
| 203 | 0.697 |
| 205 | 0.990 |
| 207 (BBB) | 0.853 |
| 208 | 0.803 |
| 209 | 0.950 |
| 215 | 0.963 |
| 220 | 0.979 |
| 223 | 0.917 |
| 230 | 0.854 |
