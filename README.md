# LLM6G
Forecasting per-sector traffic to support energy-aware radio access networks using pretrained Chronos models.

## Environment
- Create an isolated environment (e.g., `python3 -m venv .venv && source .venv/bin/activate`).
- Install dependencies from `requirements.txt` (`pip install -r requirements.txt`). Chronos inference runs on device; GPU is optional.

## Goal and approach
- We forecast traffic (Mbps) for each antenna sector to inform sleep-mode and energy-management policies.
- The pipeline relies on zero-shot Chronos family models (Chronos-T5, Chronos-Bolt, Chronos-2) without fine-tuning; we only prepare data and evaluate.

## Chronos at a glance ([1])
- Real-valued series are mean-scaled and quantized into a fixed vocabulary (4096 bins with PAD/EOS). For a context of length `C`, scaling uses `m = 0` and `s = (1/C) * sum_{t=1..C} |x_t|`, so `x_t` becomes `x_t / s`.
- Quantization maps each scaled value to a bin ID: `q(x) = j` when `b_{j-1} <= x < b_j`, and dequantization uses the bin center `d(j) = c_j`. Uniform bins are placed between `c_1` and `c_B` (Chronos uses `c_1 = -15`, `c_B = +15`).
- Models are standard transformers (Chronos-T5 is encoder–decoder; Chronos-Bolt/Chronos-2 are decoder-only) trained with categorical cross-entropy on the next token,
  `L = - sum_{h=1..H+1} log p_theta(z_{C+h} | z_{1:C+h-1})`, so forecasting is regression-via-classification rather than parametric likelihood.
- During inference we autoregressively sample from `p_theta(z_{C+h} | z_{1:C+h-1})`, dequantize, and unscale. Point forecasts in this repo use either the 0.5-quantile returned by Bolt/Chronos-2 or the mean of `num_samples` draws for sample-based models.

## Data
- Historical dataset (`data/histo_trafic.csv`): 86 sectors, weekly measurements from June 2018 to January 2024. After preprocessing (`data/processed_trafic_original.jsonl`) contexts range from 257–286 points (mean ≈ 284).
- Synthetic instantaneous dataset (`data/histo_trafic_instant.csv`): per-sector high-frequency series generated to emulate bursty arrivals following the digital-twin traffic model of Masoudi et al. ([2]). For each 5-minute slot we compute empirical mean/variance across days, assume an interrupted Poisson process (IPP) with ON/OFF rates `tau` and `zeta`, and solve for the Poisson rate `lambda` and mean per-arrival demand `E[psi]` such that:
  ```
  E[U] = lambda * tau/(tau + zeta) * T
  Var(U) ≈ lambda * tau/(tau + zeta) * T * (1 + 2*lambda*zeta/(tau + zeta)^2)
  E[Psi] = E[U] * E[psi]
  Var(Psi) = E[U] * Var(psi) + Var(U) * (E[psi])^2
  ```
  where `U` is the number of arrivals in window `T` and `Psi` the aggregated rate. This yields per-second traffic sequences (≈49k–58k points per sector). A 5-sector subset lives in `data/histo_trafic_instant_short.csv` for quicker experiments.
- `scripts/dataprocessing.py` normalizes timestamps (French dates → ISO), groups by sector, and writes one context/target pair per sector to JSONL for downstream evaluation.

## Evaluation pipeline
- `scripts/single_eval.py` loads a Chronos pipeline, feeds each sector’s full history as context, and forecasts the final step (prediction length 1). The Chronos library internally truncates to each model’s maximum context window.
- For sample-based models we draw `num_samples=32` forecasts and average; for quantile-returning models we take the median. We report RMSE across sectors.

## Results
- Historical weekly data (`results/summary_hist_trafic_original.csv`, 86 sectors): best RMSE from `amazon/chronos-bolt-mini` (6.72), slightly ahead of larger variants.
- Synthetic instantaneous subset (`results/summary_hist_trafic_instant_short.csv`, 5 sectors): best RMSE from `amazon/chronos-bolt-base` (5.14). Parameter count does not monotonically improve accuracy.

## References
- [1] A. F. Ansari et al., “Chronos: Learning the Language of Time Series,” TMLR 2024. (`sources/Chronos.pdf`)
- [2] M. Masoudi et al., “Digital Twin Assisted Risk-Aware Sleep Mode Management Using Deep Q-Networks,” arXiv:2208.14380, 2022. (`sources/KTH.pdf`)
