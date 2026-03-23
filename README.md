# LLM6G
Codebase for the paper:
**Shift-Aware Forecast-to-Control on TIM Milan: A Public Telecom Benchmark for Zero-Shot and Trainable Models**

## What this repo is
- This repo is the reproducible codebase for our FINE 2026 paper submission.
- The paper studies a **shift-aware forecast-to-control pipeline** on public telecom traffic.
- The benchmark is built from the public **TIM Milan** dataset.
- The main comparison is:
  `seasonal_naive`, `LSTM`, `DeepAR`, and `Chronos-2` in zero-shot mode.

## Paper claim
- Traffic control should account for the **next change of distribution**, not only point accuracy.
- The pipeline in this repo is:
  forecast probabilistic traffic ->
  predict the first break point ->
  compute a pre-break safe ceiling ->
  hold the control policy over that interval ->
  rerun after the predicted break.
- The practical question is:
  can a zero-shot foundation model enter that loop and remain operationally credible against trained baselines?

## Main artifact
- Public benchmark:
  TIM Milan `InternetTraffic` at native `10min` cadence.
- Full clean benchmark:
  `10,000` raw grid cells,
  `8,883` fully observed cells kept,
  `1,117` incomplete cells dropped.
- Trainable subset:
  `200` complete cells selected with a neutral deterministic rule.
- Current main reproduced run:
  `context = 144`,
  `horizon = 48`,
  `70 / 10 / 20` chronological split,
  `seed = 42`.

## Repository map
- `src/`
  pipeline code, forecasting models, evaluation, experiment runner
- `data/`
  dataset builders for TIM Milan and the trainable subset
- `results/experiments/`
  full machine-readable outputs of experiment runs
- `results/plots/readme/`
  stable figures used in the README and paper
- `notebooks/experiment_report.ipynb`
  maintained report notebook for a finished run
- `notebooks/legacy/`
  archived exploratory notebooks kept outside the paper-facing workflow
- `experiment_notes.md`
  archival notes, older experiments, and meeting-log material

## Setup
- Create an environment:
  `python3 -m venv .venv && source .venv/bin/activate`
- Install dependencies:
  `pip install -r requirements.txt`
- Chronos-2 inference can run on CPU.
- The paper PDF in `paper/` requires a local TeX installation.

## Data
- Dataset source:
  Gianni Barlacchi et al., *A multi-source dataset of urban life in the city of Milan and the Province of Trentino*, *Scientific Data*, 2015.
- Raw telecom files are accessed through the O-RAN SC mirror/reference page.
- In this repo, we use only the `InternetTraffic` field.
- The processed series are **Milan grid cells**, not radio sectors.
- Values are normalized public telecom activity proxies, not literal Mbps counters.

### Build the full TIM Milan benchmark
- Command:
  `python data/build_tim_milan_dataset.py`
- Outputs:
  `data/data_tim_milan_10min.csv`
  `data/data_tim_milan_10min_metadata.csv`
  `data/data_tim_milan_10min_dropped_cells.csv`
- Builder policy:
  keep the native `10min` cadence,
  align the full city-wide grid,
  keep all fully observed cells,
  drop only incomplete cells.

### Build the trainable subset
- Command:
  `python data/build_tim_milan_trainable_subset.py`
- Outputs:
  `data/data_tim_milan_10min_trainable_200.csv`
  `data/data_tim_milan_10min_trainable_200_metadata.csv`
- Selection rule:
  sort complete cells by ascending `square_id`,
  keep the first `200`.
- This rule is neutral and reproducible.

## Run the published experiment
- Main command:
  `python src/run_experiment.py \
  --overwrite \
  --with-tau-calibration \
  --data-path data/data_tim_milan_10min_trainable_200.csv \
  --context-length 144 \
  --horizon 48 \
  --models lstm,deepar,chronos2,seasonal_naive \
  --max-iterations 10000 \
  --patience-iterations 1000 \
  --validate-every 5000`
- This runs:
  `prepare_data -> train -> forecast_eval -> system_eval -> cp_sweep -> tau_calibration`
- `seasonal_naive`:
  copy the value from the same time one day earlier,
  then add a simple history-derived upper margin for `q95`
- Main experiment directory:
  `results/experiments/data_tim_milan_10min_trainable_200_ctx144_h48/q50_q95_seed42`
- The pipeline is cadence-flexible.
- The same runner also works on other regular timestamp grids.

## Evaluation logic
- Forecast outputs:
  `q50` and `q95`
- Control logic:
  detect `tau_pred` on `q50`,
  compute `safe_ceiling = max(q95[:tau_pred])`,
  evaluate the resulting pre-break control interval
- Forecast metrics:
  `RMSE_q50`, `Pinball_q50`, `Pinball_q95`, `Coverage_q95`, interval width
- System metrics:
  `MAE_CP`, tolerance hit rate, `coverage_rate`, `sharpness`
- The control interpretation is:
  `coverage_rate` measures protection under the ceiling
  `sharpness` measures how conservative the ceiling is

## Main reproduced results
- `LSTM` is strongest overall on the current Milan `200`-cell benchmark.
- `Chronos-2` is the main zero-shot baseline.
- `Chronos-2` stays close to `DeepAR` on system metrics while avoiding Milan-specific retraining.
- Zero-shot Chronos-2 is not the best model on this run,
  but it is a credible no-retraining operating point inside the full control loop.

![Forecast evaluation on Milan test windows](results/experiments/data_tim_milan_10min_trainable_200_ctx144_h48/q50_q95_seed42/reports/forecast_eval_test.png)

![System evaluation on Milan test windows](results/experiments/data_tim_milan_10min_trainable_200_ctx144_h48/q50_q95_seed42/reports/system_eval_test.png)

Per-model pipeline examples on the same saved Milan test window:

![LSTM pipeline example](results/experiments/data_tim_milan_10min_trainable_200_ctx144_h48/q50_q95_seed42/reports/example_windows_system_eval_lstm_test.png)

![DeepAR pipeline example](results/experiments/data_tim_milan_10min_trainable_200_ctx144_h48/q50_q95_seed42/reports/example_windows_system_eval_deepar_test.png)

![Chronos-2 pipeline example](results/experiments/data_tim_milan_10min_trainable_200_ctx144_h48/q50_q95_seed42/reports/example_windows_system_eval_chronos2_test.png)

![Seasonal-naive pipeline example](results/experiments/data_tim_milan_10min_trainable_200_ctx144_h48/q50_q95_seed42/reports/example_windows_system_eval_seasonal_naive_test.png)

### Interpretation and failure modes
- The saved `144 / 48` example windows show that the deep baselines do not fail in the same way.
- `LSTM` fails mainly through **level instability**:
  on some Milan cells it shifts the whole future path far too high,
  and the associated `q95` becomes extremely conservative.
  In practice this yields very high coverage but poor sharpness and early predicted change points.
- A likely reason is architectural:
  the current `LSTM` forecaster is a direct encoder-to-horizon map with no explicit scale normalization,
  no explicit daily-seasonality structure,
  and no autoregressive correction step.
  On a heterogeneous multi-series dataset this can make the learned level fragile across cells and regimes.
- `DeepAR` fails in the opposite direction:
  its forecast paths in the saved examples are too flat and too low,
  and its `q95` often under-covers the realized future.
  That produces safe ceilings that are too low and change-point errors that are operationally unsafe.
- A likely reason is the train/eval mismatch:
  `DeepAR` is optimized with teacher forcing during training,
  but is used autoregressively at forecast time.
  Small one-step errors then feed back into later steps,
  which can flatten the trajectory and collapse upper-tail coverage.
- The comparison with `seasonal_naive` is important:
  on strongly daily-periodic windows,
  a simple lag-1-day baseline can preserve the local shape better than the learned deep models.
  This is an informative negative result, not just a bad run:
  on this benchmark, generic recurrent models can be less reliable than a strong seasonal prior when calibration and control safety matter.
- For the paper, the main takeaway is:
  aggregate metrics alone are not enough.
  The example windows reveal two distinct failure modes,
  `LSTM` over-predicting with over-wide ceilings and `DeepAR` under-predicting with under-covered ceilings,
  and both matter directly for forecast-to-control deployment.

## Secondary analyses
- `cp_sweep`
  validation sweep of the PELT change-point detector
- `tau_calibration`
  post-hoc correction of predicted break points
- In the current paper story:
  CP tuning stays in the main pipeline,
  tau calibration is a mixed exploratory result and not part of the deployed main method.

## Paper
- LaTeX sources:
  `paper/main.tex`
  `paper/sections/`
  `paper/figures/`
  `paper/references.bib`
- Compile from `paper/` with:
  `latexmk -pdf main.tex`
- Default paper mode is anonymized submission.
- Camera-ready authors can be restored through the toggle in `paper/main.tex`.

## Notes
- `README.md` is the codebase companion of the paper.
- `experiment_notes.md` keeps older context and historical notes.
- Legacy private-data and augmented-data experiments are kept out of the main paper story.

## References
- IEEE FINE 2026 CFP:
  <https://www.ieee-fine.org/2026/>
- TIM Milan dataset paper:
  <https://doi.org/10.1038/sdata.2015.55>
- O-RAN SC dataset mirror/reference page:
  <https://lf-o-ran-sc.atlassian.net/wiki/spaces/SIM/pages/13435002/Simulated+datasets>
- Chronos:
  <https://arxiv.org/abs/2403.07815>
- Chronos-2:
  <https://arxiv.org/abs/2510.15821>
- DeepAR:
  <https://arxiv.org/abs/1704.04110>
