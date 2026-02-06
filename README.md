# TubeCoreEEG
# RegenSpec

Physics-inspired EEG power spectral density (PSD) fitting framework.  
Double-Gaussian cores (theta + alpha) + power-law background, originally motivated by QCD flux tube energy profiles.

Fits raw EEG data across domains:
- Epilepsy intervention (pre/post SIFT regen → sharp alpha, low χ²_red)
- Healthy BCI motor imagery (task suppression → low fp_alpha, mid-high RMS)

## Model

$$    
P(f) = A_\theta \exp\left(-\frac{(f - f_{p\theta})^2}{2\sigma_\theta^2}\right) + A_\alpha \exp\left(-\frac{(f - f_{p\alpha})^2}{2\sigma_\alpha^2}\right) + (f + 0.1)^\alpha + \text{offset}
    $$

## Key Results (BCI ds003190, 384 runs, Cz channel)

- 363/384 successful fits (~94%)
- fp_theta ~0.5–1 Hz (low-freq dominance)
- fp_alpha ~5 Hz (pinned, task suppression), occasionally higher
- χ²_red 0.02–0.4 (excellent for raw data)
- RMS freq 11–23 Hz (mid-high power weight)

Full metrics in `./bci_fits/`

## Install & Run

```bash
pip install -r requirements.txt
python src/fit_dataset.py
