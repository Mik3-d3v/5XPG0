CV (5 patient folds on DS1, pooled out-of-fold), epochs=20, seeds=[0, 1, 2], batch=256, lr=0.003, wd=0.2, hidden=64, timesteps=25, beta=0.9, encoding=direct, use_rr=True, morph=diff, augment=True

| metric (mean +- std over seeds) | RSNN |
|---|---|
| macro_f1 | 0.633 +- 0.007 |
| accuracy | 0.917 +- 0.003 |
| se_N | 0.941 +- 0.003 |
| se_S | 0.432 +- 0.017 |
| se_V | 0.744 +- 0.010 |
| ppv_N | 0.984 +- 0.000 |
| ppv_S | 0.176 +- 0.010 |
| ppv_V | 0.639 +- 0.020 |

| record | RSNN acc |
|---|---|
| 101 | 0.996 |
| 106 | 0.935 |
| 108 | 0.950 |
| 109 (BBB) | 0.968 |
| 112 | 0.998 |
| 114 | 0.993 |
| 115 | 0.998 |
| 116 | 0.996 |
| 118 (BBB) | 0.970 |
| 119 | 0.988 |
| 122 | 0.975 |
| 124 (BBB) | 0.953 |
| 201 | 0.804 |
| 203 | 0.661 |
| 205 | 0.989 |
| 207 (BBB) | 0.848 |
| 208 | 0.757 |
| 209 | 0.920 |
| 215 | 0.910 |
| 220 | 0.991 |
| 223 | 0.918 |
| 230 | 0.769 |
