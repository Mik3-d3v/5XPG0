CV (5 patient folds on DS1, pooled out-of-fold), epochs=20, seeds=[0, 1, 2], batch=256, lr=0.003, wd=0.2, hidden=64, timesteps=25, beta=0.9, encoding=direct, use_rr=True, morph=diff, augment=True

| metric (mean +- std over seeds) | RSNN |
|---|---|
| macro_f1 | 0.684 +- 0.002 |
| accuracy | 0.943 +- 0.000 |
| se_N | 0.972 +- 0.000 |
| se_S | 0.431 +- 0.018 |
| se_V | 0.725 +- 0.011 |
| ppv_N | 0.979 +- 0.001 |
| ppv_S | 0.262 +- 0.002 |
| ppv_V | 0.779 +- 0.011 |

| record | RSNN acc |
|---|---|
| 101 | 0.998 |
| 106 | 0.904 |
| 108 | 0.931 |
| 109 (BBB) | 0.994 |
| 112 | 0.999 |
| 114 | 0.995 |
| 115 | 1.000 |
| 116 | 0.990 |
| 118 (BBB) | 0.963 |
| 119 | 0.999 |
| 122 | 1.000 |
| 124 (BBB) | 0.968 |
| 201 | 0.845 |
| 203 | 0.732 |
| 205 | 0.993 |
| 207 (BBB) | 0.851 |
| 208 | 0.846 |
| 209 | 0.957 |
| 215 | 0.988 |
| 220 | 0.978 |
| 223 | 0.899 |
| 230 | 0.964 |
