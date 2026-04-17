| Model | Params | Annual Infer FLOPs / Market | Annual Retrain FLOPs (10 mkts) | Annual Total FLOPs (10 mkts) | Annual GPU Joules (10 mkts) | Annual CPU Joules (10 mkts) | Rel. Sharp. (covered) | Δ vs Naive |
|---|---|---|---|---|---|---|---|---|
| TFT | 1.83e+06 | 7.37e+15 | 6.80e+16 | 1.42e+17 | 1.84e+05 | 1.42e+08 | 0.378 | 0.055 |
| LSTM | 211,552 | 8.54e+14 | 2.95e+16 | 3.81e+16 | 4.95e+04 | 3.81e+07 | 0.549 | -0.115 |
| DeepAR | 215,202 | 8.69e+14 | 4.81e+16 | 5.68e+16 | 7.38e+04 | 5.68e+07 | 0.219 | 0.215 |
| Chronos-2 | 1.20e+08 | 3.03e+16 | 0.00e+00 | 3.03e+17 | 3.94e+05 | 3.03e+08 | 0.437 | -0.003 |
| Seasonal Naive | — | 1.14e+10 | 0.00e+00 | 1.14e+11 | 1.48e-01 | 1.14e+02 | 0.433 | 0.000 |
