### 1. Latency distribution, fast firmware [ms/sample]

| model | seed | n | mean | p5 | median | p95 | min | max | bit-exact |
|---|---|---|---|---|---|---|---|---|---|
| ANN | 0 | 10000 | 13.62 | 8.32 | 13.62 | 18.70 | 5.56 | 25.57 | 10000/10000 |
| ANN | 1 | 10000 | 13.62 | 8.28 | 13.62 | 18.70 | 5.46 | 25.61 | 10000/10000 |
| SNN | 0 | 10000 | 57.06 | 44.28 | 57.61 | 66.68 | 22.96 | 78.50 | 10000/10000 |
| SNN | 1 | 10000 | 57.20 | 44.45 | 57.71 | 66.92 | 23.78 | 79.72 | 10000/10000 |

### 2. Speed-up of the optimisations (same images)

| model | seed | n | baseline [ms] | fast [ms] | speed-up | per-image min / max |
|---|---|---|---|---|---|---|
| ANN | 0 | 1000 | 135.79 | 12.96 | 10.48× | 5.7× / 23.9× |
| ANN | 1 | 1000 | 135.79 | 12.97 | 10.47× | 5.7× / 24.3× |
| SNN | 0 | 1000 | 402.21 | 56.47 | 7.12× | 3.2× / 15.5× |
| SNN | 1 | 1000 | 403.08 | 56.59 | 7.12× | 3.1× / 15.3× |

### 3. Cost model of the fast kernels (least squares, all images)

| model | seed | cycles ≈ | R² | fixed part [ms] | share of mean |
|---|---|---|---|---|---|
| ANN | 0 | 7,571 · nonzero pixel + 1,198 · nonzero hidden + 119,210 | 1.0000 | 1.19 | 9 % |
| ANN | 1 | 7,575 · nonzero pixel + 1,188 · nonzero hidden + 119,713 | 1.0000 | 1.20 | 9 % |
| SNN | 0 | 3,450 · input change + 124 · hidden spike + 2,889,887 | 0.9952 | 28.90 | 51 % |
| SNN | 1 | 3,455 · input change + 148 · hidden spike + 2,884,259 | 0.9957 | 28.84 | 50 % |

### 4. Paired per-image comparison, fast firmware (same test image)

| seed | n | SNN/ANN latency: mean ratio | median | min | max | corr(ANN, SNN latency) | images where SNN is faster |
|---|---|---|---|---|---|---|---|
| 0 | 10000 | 4.19 | 4.27 | 2.23 | 7.72 | 0.825 | 0 |
| 1 | 10000 | 4.20 | 4.27 | 2.31 | 7.75 | 0.828 | 0 |

### 5. Predictions: ANN vs SNN (fast firmware, same images)

| seed | n | both right | only ANN right | only SNN right | both wrong | same prediction |
|---|---|---|---|---|---|---|
| 0 | 10000 | 9637 | 58 | 135 | 170 | 9775 |
| 1 | 10000 | 9617 | 56 | 140 | 187 | 9768 |

Per class (mean over seeds):

| digit | ANN acc [%] | SNN acc [%] | ANN latency [ms] | SNN latency [ms] |
|---|---|---|---|---|
| 0 | 98.83 | 99.23 | 16.49 | 62.88 |
| 1 | 98.77 | 99.21 | 8.77 | 44.95 |
| 2 | 96.61 | 97.43 | 14.92 | 59.87 |
| 3 | 97.08 | 98.42 | 14.64 | 59.61 |
| 4 | 97.10 | 97.76 | 12.93 | 56.12 |
| 5 | 95.57 | 96.97 | 13.91 | 58.99 |
| 6 | 97.18 | 97.34 | 14.40 | 58.48 |
| 7 | 96.30 | 97.18 | 12.15 | 53.82 |
| 8 | 95.48 | 96.20 | 15.56 | 61.81 |
| 9 | 95.14 | 96.43 | 13.24 | 56.85 |

### 6. ANN kernels: ours baseline, muRISCV-NN, ours fast (same images) [ms/image]

| seed | n | ANN baseline | ANN muRISCV-NN | ANN fast | SNN fast | muRISCV-NN vs baseline | fast vs muRISCV-NN | images where SNN fast < muRISCV-NN |
|---|---|---|---|---|---|---|---|---|
| 0 | 10000 | 135.79 (n=1000) | 76.62 (min 76.61, max 76.62) | 13.62 | 57.06 | 1.77× faster | 5.62× faster | 9998/10000 |
| 1 | 10000 | 135.79 (n=1000) | 76.62 (min 76.61, max 76.62) | 13.62 | 57.20 | 1.77× faster | 5.62× faster | 9998/10000 |

### 7. Predictions: muRISCV-NN vs our int8 ANN (same images)

| seed | n | muRISCV-NN acc | ours acc | both right | only ours right | only muRISCV-NN right | both wrong | same prediction | McNemar p (exact) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 10000 | 96.92 % | 96.95 % | 9687 | 8 | 5 | 300 | 9983 | 0.581 |
| 1 | 10000 | 96.78 % | 96.73 % | 9664 | 9 | 14 | 313 | 9974 | 0.405 |

### 8. Cycles per multiply-accumulate (784-128-10: 101,632 dense MACs)

| seed | ours baseline, per dense MAC | muRISCV-NN, per dense MAC | ours fast, per executed MAC (cost-model slope / 128) | ours fast, per dense MAC |
|---|---|---|---|---|
| 0 | 133.6 | 75.4 | 59.2 | 13.4 |
| 1 | 133.6 | 75.4 | 59.2 | 13.4 |
