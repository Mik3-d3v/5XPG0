### 1. Latency distribution, fast firmware [ms/sample]

| model | seed | n | mean | p5 | median | p95 | min | max | bit-exact |
|---|---|---|---|---|---|---|---|---|---|
| ANN | 0 | 49273 | 4.88 | 3.75 | 4.96 | 5.70 | 0.85 | 5.96 | 49273/49273 |
| ANN | 1 | 49273 | 4.87 | 3.74 | 4.95 | 5.70 | 0.81 | 5.93 | 49273/49273 |
| ANN | 2 | 49273 | 4.87 | 3.73 | 4.95 | 5.70 | 0.80 | 5.92 | 49273/49273 |
| SNN | 0 | 49273 | 8.34 | 6.93 | 8.36 | 9.55 | 3.64 | 10.29 | 49273/49273 |
| SNN | 1 | 49273 | 8.28 | 6.83 | 8.31 | 9.51 | 3.57 | 10.06 | 49273/49273 |
| SNN | 2 | 49273 | 8.28 | 6.87 | 8.29 | 9.51 | 3.70 | 10.14 | 49273/49273 |

### 2. Speed-up of the optimisations (same images)

| model | seed | n | baseline [ms] | fast [ms] | speed-up | per-image min / max |
|---|---|---|---|---|---|---|
| ANN | 0 | 49273 | 11.71 | 4.88 | 2.40× | 2.0× / 13.9× |
| ANN | 1 | 49273 | 11.70 | 4.87 | 2.40× | 2.0× / 14.5× |
| ANN | 2 | 49273 | 11.70 | 4.87 | 2.41× | 2.0× / 14.6× |
| SNN | 0 | 49273 | 13.48 | 8.34 | 1.62× | 1.4× / 3.6× |
| SNN | 1 | 49273 | 13.45 | 8.28 | 1.62× | 1.4× / 3.7× |
| SNN | 2 | 49273 | 13.44 | 8.28 | 1.62× | 1.4× / 3.6× |

### 3. Cost model of the fast kernels (least squares, all images)

| model | seed | cycles ≈ | R² | fixed part [ms] | share of mean |
|---|---|---|---|---|---|
| ANN | 0 | 3,921 · nonzero input + 818 · nonzero hidden + 48,902 | 0.9999 | 0.49 | 10 % |
| ANN | 1 | 3,917 · nonzero input + 821 · nonzero hidden + 49,373 | 0.9999 | 0.49 | 10 % |
| ANN | 2 | 3,920 · nonzero input + 817 · nonzero hidden + 49,097 | 0.9999 | 0.49 | 10 % |
| SNN | 0 | 4,471 · nonzero input + 246 · hidden spike + 353,327 | 0.9434 | 3.53 | 42 % |
| SNN | 1 | 4,549 · nonzero input + 286 · hidden spike + 338,139 | 0.9556 | 3.38 | 41 % |
| SNN | 2 | 4,466 · nonzero input + 312 · hidden spike + 345,332 | 0.9634 | 3.45 | 42 % |

### 4. Paired per-image comparison, fast firmware (same test image)

| seed | n | SNN/ANN latency: mean ratio | median | min | max | corr(ANN, SNN latency) | images where SNN is faster |
|---|---|---|---|---|---|---|---|
| 0 | 49273 | 1.71 | 1.70 | 1.52 | 4.52 | 0.958 | 0 |
| 1 | 49273 | 1.70 | 1.69 | 1.51 | 4.53 | 0.961 | 0 |
| 2 | 49273 | 1.70 | 1.69 | 1.52 | 4.65 | 0.963 | 0 |

### 5. Predictions: ANN vs SNN (fast firmware, same images)

| seed | n | both right | only ANN right | only SNN right | both wrong | same prediction |
|---|---|---|---|---|---|---|
| 0 | 49273 | 45744 | 316 | 605 | 2608 | 48225 |
| 1 | 49273 | 45853 | 321 | 556 | 2543 | 48264 |
| 2 | 49273 | 45858 | 289 | 601 | 2525 | 48256 |

Per class (mean over seeds):

| class | ANN acc [%] | SNN acc [%] | ANN latency [ms] | SNN latency [ms] |
|---|---|---|---|---|
| N | 96.80 | 97.07 | 4.82 | 8.22 |
| S | 17.30 | 23.40 | 5.17 | 8.69 |
| V | 93.34 | 94.93 | 5.48 | 9.18 |

### 6. ANN kernels: ours baseline, muRISCV-NN, ours fast (same beats) [ms/beat]

| seed | n | ANN baseline | ANN muRISCV-NN | ANN fast | SNN fast | muRISCV-NN vs baseline | fast vs muRISCV-NN | beats where SNN fast < muRISCV-NN |
|---|---|---|---|---|---|---|---|---|
| 0 | 49273 | 11.71 (n=49273) | 7.05 (min 7.05, max 7.06) | 4.88 | 8.34 | 1.66× faster | 1.44× faster | 3231/49273 |
| 1 | 49273 | 11.70 (n=49273) | 7.05 (min 7.05, max 7.06) | 4.87 | 8.28 | 1.66× faster | 1.45× faster | 3968/49273 |
| 2 | 49273 | 11.70 (n=49273) | 7.05 (min 7.05, max 7.06) | 4.87 | 8.28 | 1.66× faster | 1.45× faster | 3729/49273 |

### 7. Predictions: muRISCV-NN vs our int8 ANN (same beats)

| seed | n | muRISCV-NN acc | ours acc | both right | only ours right | only muRISCV-NN right | both wrong | same prediction | McNemar p (exact) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 49273 | 93.58 % | 93.48 % | 46040 | 20 | 69 | 3144 | 49173 | 0.000 |
| 1 | 49273 | 93.77 % | 93.71 % | 46158 | 16 | 44 | 3055 | 49200 | 0.000 |
| 2 | 49273 | 93.72 % | 93.66 % | 46140 | 7 | 37 | 3089 | 49220 | 0.000 |

### 8. Cycles per multiply-accumulate (129-64-3: 8,448 dense MACs)

| seed | ours baseline, per dense MAC | muRISCV-NN, per dense MAC | ours fast, per executed MAC (cost-model slope / 64) | ours fast, per dense MAC |
|---|---|---|---|---|
| 0 | 138.6 | 83.5 | 61.3 | 57.8 |
| 1 | 138.5 | 83.5 | 61.2 | 57.6 |
| 2 | 138.5 | 83.5 | 61.3 | 57.6 |
