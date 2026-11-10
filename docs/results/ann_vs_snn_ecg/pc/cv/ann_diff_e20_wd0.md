CV (5 patient folds on DS1, pooled out-of-fold), epochs=20, seeds=[0, 1, 2], batch=64, lr=0.001, wd=0.0, hidden=64, timesteps=25, beta=0.9, encoding=direct, use_rr=True, morph=diff, augment=False

| metric (mean +- std over seeds) | ANN |
|---|---|
| macro_f1 | 0.641 +- 0.011 |
| accuracy | 0.929 +- 0.003 |
| se_N | 0.956 +- 0.001 |
| se_S | 0.297 +- 0.010 |
| se_V | 0.762 +- 0.035 |
| ppv_N | 0.979 +- 0.000 |
| ppv_S | 0.133 +- 0.010 |
| ppv_V | 0.781 +- 0.009 |

| record | ANN acc |
|---|---|
| 101 | 0.998 |
| 106 | 0.899 |
| 108 | 0.895 |
| 109 (BBB) | 0.992 |
| 112 | 0.998 |
| 114 | 0.994 |
| 115 | 1.000 |
| 116 | 0.994 |
| 118 (BBB) | 0.968 |
| 119 | 1.000 |
| 122 | 1.000 |
| 124 (BBB) | 0.971 |
| 201 | 0.868 |
| 203 | 0.788 |
| 205 | 0.982 |
| 207 (BBB) | 0.838 |
| 208 | 0.896 |
| 209 | 0.905 |
| 215 | 0.785 |
| 220 | 0.973 |
| 223 | 0.931 |
| 230 | 0.884 |
