# LLM6G
Forecasting per-sector traffic to support energy-aware radio access networks using pretrained Chronos models.

## Environment
- Create an isolated environment (e.g., `python3 -m venv .venv && source .venv/bin/activate`).
- Install dependencies from `requirements.txt` (`pip install -r requirements.txt`). Chronos inference runs on device; GPU is optional.

## Goal and approach
- We forecast traffic (Mbps) for each antenna sector to inform sleep-mode and energy-management policies.
- The pipeline relies on zero-shot Chronos family models (Chronos-T5, Chronos-Bolt, Chronos-2) without fine-tuning; we only prepare data and evaluate.

## Overall view on the Chronos models
- The Chronos models are foundational models for time series. ([1]). They are pre-trained on an extremely large time-series dataset (here put the approximate size). Its architecture is the same as an LLM: transformer-based for long-term attention, parallelized training, and scalable parameter count, using a decoder-only structure (Chronos-T5 is encoder–decoder; Chronos-Bolt/Chronos-2 are decoder-only). Tokenization is the main difference compared to text generation tasks. Here the input space is a continuous 1-D time series.
The continuous input space is discretized into tokens, called bins.
- For each input, we take a context of size `C` and predict a horizon of size `H`. Real-valued series are mean-scaled and quantized into a fixed vocabulary (4096 bins with PAD/EOS). The context vector is used to perform starndard-deviation-normalization (mean = 0) of the input. Scaling uses `m = 0` and `s = (1/C) * sum_{t=1..C} |x_t|`, so `x_t` becomes `x_t / s`. 
- Then the input is discretized. Quantization maps each scaled value to a bin ID: `q(x) = j` when `b_{j-1} <= x < b_j`, and dequantization uses the bin center `d(j) = c_j`. Uniform bins are placed between `c_1` and `c_B` (Chronos uses `c_1 = -15`, `c_B = +15`). 
- At this point we have the same setting as an LLM input, namely a sequence of tokens. We pad the input sequence if it is shorter than the context size, and we truncate by taking only the last values if it is longer.
- We then compute the distributions of the output sequence autoregressively. This means we use the C context tokens to get the forecast distribution of Xc+1; then we append the sampled token for `z_{C+1}` to the input sequence and use it to predict `z_{C+1}`, and so on, until the full prediction horizon is generated. `L = - sum_{h=1..H+1} log p_theta(z_{C+h} | z_{1:C+h-1})`.
- Forecasting is regression-via-classification. The predictions are distributions: during training, Chronos statistically learns the “general” distribution of time-series data (of every kind! e.g. finance, environmental, energy...) by minimizing cross-entropy between Chronos’s predicted token distribution and the “real-world” distribution represented in the large pretraining dataset.
- Now, regarding selection of the next token: we do not take the token with highest probability (argmax), unless we want deterministic decoding. Instead, Chronos uses the predicted probability distribution over bins to compute quantiles. From these quantiles, the model produces the median forecast and prediction intervals. Concretely, quantiles are obtained by inverting the discrete CDF derived from the token probabilities, giving a median and confidence intervals for each autoregressive step.
- During inference we autoregressively sample from `p_theta(z_{C+h} | z_{1:C+h-1})`, dequantize, and unscale. Point forecasts in this repo use either the 0.5-quantile returned by Bolt/Chronos-2 or the mean of `num_samples` draws for sample-based models.

## Data
- For our project, we want to predict the consumption of Mbps in antenna sectors to enable intelligent energy usage. The traffic on different antennas varies during the day, week, and months. There is not always the same number of people using them depending on the time. To make the system's energy consumption intelligent, we try to predict the Mbps consumption over time for the antennas.
- We have a dataset (`data/histo_trafic.csv`) of scalar values (Mbps) for 86 antennas, taken every week between June 2018 and January 2024, with between 257–286 scalar values for each antenna.
- `scripts/dataprocessing.py` normalizes timestamps (French dates → ISO), groups by sector, and writes one context/target pair per sector to JSONL for downstream evaluation. After preprocessing (`data/processed_trafic_original.jsonl`) contexts range from 257–286 points (mean ≈ 284).
- The first step is to expand this dataset using data augmentation techniques to create an instantaneous dataset. The idea, in a nutshell, is to compute statistical features from the dataset and use them to estimate the number of users during each period, which is then used to augment the data:
- Synthetic instantaneous dataset (`data/histo_trafic_instant.csv`): per-sector high-frequency series generated to emulate bursty arrivals following the digital-twin traffic model of Masoudi et al. ([2]). For each 5-minute slot (original dataset) we compute empirical mean/variance across days, assume an interrupted Poisson process (IPP) with ON/OFF rates `tau` and `zeta`, and solve for the Poisson rate `lambda` and mean per-arrival demand `E[psi]` such that:
  ```
  E[U] = lambda * tau/(tau + zeta) * T
  Var(U) ≈ lambda * tau/(tau + zeta) * T * (1 + 2*lambda*zeta/(tau + zeta)^2)
  E[Psi] = E[U] * E[psi]
  Var(Psi) = E[U] * Var(psi) + Var(U) * (E[psi])^2
  ```
  where `U` is the number of arrivals in window `T` and `Psi` the aggregated rate. 
- This yields per-second traffic sequences (≈49k–58k points per sector). A 5-sector subset lives in `data/histo_trafic_instant_short.csv` for quicker experiments. We then obtain a much larger dataset of size: 86 (antennas) times ~55k (augmented scalar values).
- Now, we compute the prediction of the last value in each antenna’s dataset using all previous values as context. The context (~55k tokens) is much larger than what the models can process: the maximum context sizes of the Chronos models range from 512 tokens (Chronos) to 8192 tokens (Chronos 2). Therefore, the models crop the input and only keep the most recent values (excluding the very last one) as context for predicting the final value.
- We perform the forecasting separately for each of the 86 antennas. The predictions are inherently stochastic, we repeat the experiment num_samples = 32 times and take the mean the 32 draws as the final prediction for sample based models.
- For each model in the Chronos family, we then compute the RMSE between this averaged prediction and the ground truth. Finally, we compute the average RMSE over all entries of our dataset, meaning over all antennas.

## Evaluation pipeline
- `scripts/single_eval.py` loads a Chronos pipeline, feeds each sector’s full history as context, and forecasts the final step (prediction length 1). The Chronos library internally truncates to each model’s maximum context window.
- For sample-based models we draw `num_samples=32` forecasts and average; for quantile-returning models we take the median. We report RMSE across sectors.

## Results
- Historical weekly data (`results/summary_hist_trafic_original.csv`, 86 sectors): best RMSE from `amazon/chronos-bolt-mini` (6.72).
- Synthetic instantaneous subset (`results/summary_hist_trafic_instant.csv`, 86 sectors): best RMSE from `amazon/chronos-bolt-tiny` (2.11).
- Interestingly the predicion accuracy doesn't always improve with the parameter count being higher. A reason for that might be the amount of predictions being low, the RMSE is only averaged over 86 values.
- The RMSE of the augmented dataset being three times lower than the one of the original dataset mainly is because of the amount of 0 scalar target values being higher, and those values being easier for the models to predict (lower RMSE for these targets). 
- Check `/results/evals/` for predictions and targets.

## References
- [1] A. F. Ansari et al., “Chronos: Learning the Language of Time Series,” TMLR 2024. (`sources/Chronos.pdf`)
- [2] M. Masoudi et al., “Digital Twin Assisted Risk-Aware Sleep Mode Management Using Deep Q-Networks,” arXiv:2208.14380, 2022. (`sources/KTH.pdf`)
