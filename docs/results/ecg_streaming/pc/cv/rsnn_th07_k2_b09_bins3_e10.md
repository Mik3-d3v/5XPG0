CV (5 patient folds on DS1, pooled out-of-fold), epochs=10, seeds=[0, 1, 2], batch=256, lr=0.003, wd=0.2, hidden=64, timesteps=25, beta=0.9, encoding=direct, use_rr=True, morph=diff, augment=True

| metric (mean +- std over seeds) | RSNN |
|---|---|
| macro_f1 | 0.686 +- 0.003 |
| accuracy | 0.943 +- 0.001 |
| se_N | 0.972 +- 0.001 |
| se_S | 0.399 +- 0.008 |
| se_V | 0.732 +- 0.005 |
| ppv_N | 0.976 +- 0.000 |
| ppv_S | 0.280 +- 0.007 |
| ppv_V | 0.780 +- 0.007 |

| record | RSNN acc |
|---|---|
| 101 | 0.997 |
| 106 | 0.915 |
| 108 | 0.923 |
| 109 (BBB) | 0.994 |
| 112 | 0.998 |
| 114 | 0.995 |
| 115 | 1.000 |
| 116 | 0.993 |
| 118 (BBB) | 0.960 |
| 119 | 0.999 |
| 122 | 1.000 |
| 124 (BBB) | 0.969 |
| 201 | 0.837 |
| 203 | 0.747 |
| 205 | 0.993 |
| 207 (BBB) | 0.857 |
| 208 | 0.846 |
| 209 | 0.947 |
| 215 | 0.985 |
| 220 | 0.974 |
| 223 | 0.888 |
| 230 | 0.976 |
