CV (5 patient folds on DS1, pooled out-of-fold), epochs=20, seeds=[0, 1, 2], batch=64, lr=0.001, wd=0.05, hidden=64, timesteps=25, beta=0.9, encoding=direct, use_rr=True, morph=diff, augment=False

| metric (mean +- std over seeds) | ANN |
|---|---|
| macro_f1 | 0.646 +- 0.007 |
| accuracy | 0.931 +- 0.002 |
| se_N | 0.957 +- 0.000 |
| se_S | 0.304 +- 0.008 |
| se_V | 0.769 +- 0.024 |
| ppv_N | 0.980 +- 0.000 |
| ppv_S | 0.139 +- 0.008 |
| ppv_V | 0.786 +- 0.006 |

| record | ANN acc |
|---|---|
| 101 | 0.998 |
| 106 | 0.901 |
| 108 | 0.901 |
| 109 (BBB) | 0.993 |
| 112 | 0.998 |
| 114 | 0.994 |
| 115 | 1.000 |
| 116 | 0.994 |
| 118 (BBB) | 0.969 |
| 119 | 1.000 |
| 122 | 1.000 |
| 124 (BBB) | 0.970 |
| 201 | 0.871 |
| 203 | 0.791 |
| 205 | 0.983 |
| 207 (BBB) | 0.839 |
| 208 | 0.903 |
| 209 | 0.907 |
| 215 | 0.797 |
| 220 | 0.974 |
| 223 | 0.930 |
| 230 | 0.879 |
