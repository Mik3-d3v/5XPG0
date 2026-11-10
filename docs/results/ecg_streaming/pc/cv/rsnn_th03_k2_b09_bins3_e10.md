CV (5 patient folds on DS1, pooled out-of-fold), epochs=10, seeds=[0, 1, 2], batch=256, lr=0.003, wd=0.2, hidden=64, timesteps=25, beta=0.9, encoding=direct, use_rr=True, morph=diff, augment=True

| metric (mean +- std over seeds) | RSNN |
|---|---|
| macro_f1 | 0.658 +- 0.002 |
| accuracy | 0.930 +- 0.001 |
| se_N | 0.956 +- 0.000 |
| se_S | 0.438 +- 0.005 |
| se_V | 0.738 +- 0.003 |
| ppv_N | 0.981 +- 0.001 |
| ppv_S | 0.211 +- 0.001 |
| ppv_V | 0.703 +- 0.005 |

| record | RSNN acc |
|---|---|
| 101 | 0.998 |
| 106 | 0.916 |
| 108 | 0.952 |
| 109 (BBB) | 0.995 |
| 112 | 0.999 |
| 114 | 0.993 |
| 115 | 0.999 |
| 116 | 0.996 |
| 118 (BBB) | 0.965 |
| 119 | 0.999 |
| 122 | 1.000 |
| 124 (BBB) | 0.969 |
| 201 | 0.837 |
| 203 | 0.678 |
| 205 | 0.987 |
| 207 (BBB) | 0.853 |
| 208 | 0.792 |
| 209 | 0.941 |
| 215 | 0.951 |
| 220 | 0.985 |
| 223 | 0.925 |
| 230 | 0.808 |
