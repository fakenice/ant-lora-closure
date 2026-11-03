# Positive control (leaky by design)

The positive-control adapter is trained on the same 500 questions it is evaluated on. Its accuracy is a harness check, not a capability result.

| adapter | correct | accuracy | Wilson 95% | Δ vs base | unparseable | McNemar p |
|---|---|---|---|---|---|---|
| base | 329/500 | 65.8% | 61.5–69.8% | +0.0 pt | 1 | 1 |
| poscontrol_evaltrain | 496/500 | 99.2% | 98.0–99.7% | +33.4 pt | 0 | 4.54e-49 |
| poscontrol_shuffled | 88/500 | 17.6% | 14.5–21.2% | -48.2 pt | 2 | 5.12e-48 |

Interpretation: a near-ceiling positive control plus a near-chance shuffled control means the pipeline can detect a label-specific gain; it does not mean the model learned medicine.
