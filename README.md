# Conditional diffusion modeling for probabilistic behind-the-meter PV disaggregation

This repository contains the codebase associated with the paper:

> **Conditional diffusion modeling for probabilistic behind-the-meter PV disaggregation**  
> Marc Jené-Vinuesa, Hussain Kazmi, Mònica Aragüés-Peñalba, Andreas Sumper  
> *Energy and AI*, 2026 · [DOI: 10.1016/j.egyai.2026.100692](https://doi.org/10.1016/j.egyai.2026.100692)

The paper introduces a **conditional denoising diffusion probabilistic model (cDDPM)** for disaggregating behind-the-meter (BTM) photovoltaic (PV) generation from low-resolution smart meter data. The model generates multiple plausible daily PV profiles conditioned on net consumption and irradiance signals, providing both accurate point estimates and well-calibrated prediction intervals. Two probabilistic benchmarks — a conditional VAE (cVAE) and a quantile regression CNN (QR) — are also included.

---

## 📂 Repository Structure

```text
cddpm-probpvdis/
├── src/
│   ├── main.py          # Unified entry point — runs cddpm, cvae, or qr
│   ├── core.py          # cDDPM forward diffusion + reverse sampling
│   ├── utils.py         # Data loading, metrics, plotting utilities
│   └── models/
│       ├── unet_v2.py   # 1D U-Net with FiLM conditioning
│       ├── cvae.py      # Conditional VAE (benchmark)
│       └── quantile.py  # Quantile regression CNN (benchmark)
├── data/                # Preprocessed Ausgrid dataset (see data/README.md)
├── notebooks/
│   ├── 01_data_overview.ipynb      # Dataset description and visualisation
│   └── 02_train_and_evaluate.ipynb # End-to-end training and results
├── requirements.txt
└── CITATION.cff
```

---

## 🚀 Getting Started

### Installation (Python 3.11+)

> **Note:** Install the correct PyTorch wheel for your hardware first, then the rest.

```bash
# Option A — CPU-only
pip install torch==2.6.0

# Option B — CUDA (example: CUDA 12.4)
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0

# Then install remaining dependencies
pip install -r requirements.txt
```

### Run

```bash
cd src

# Train and evaluate the cDDPM (paper main model)
python main.py --model cddpm --seed 0

# Train and evaluate the cVAE benchmark
python main.py --model cvae

# Train and evaluate the Quantile Regression benchmark
python main.py --model qr
```

All three models share the same data loading and evaluation pipeline. Switching models only changes the `--model` flag.

### Programmatic usage

```python
from main import run, BEST_CONFIGS

# Run with the published best configuration
df_metrics = run(BEST_CONFIGS["cddpm"])

# Or customise any parameter
config = {**BEST_CONFIGS["cddpm"], "seed": 3, "n_epochs": 500}
df_metrics = run(config)
```

---

## 🗂️ Data

The `data/` folder contains preprocessed data from the publicly available [Ausgrid Solar Home Electricity Dataset](https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data) (New South Wales, Australia, July 2012 – June 2013, 30-min resolution).

A second dataset from the Netherlands, used in the paper to evaluate cross-dataset generalisation, **cannot be shared** due to privacy constraints.

See [`data/README.md`](data/README.md) for a full description of each file and the normalisation scheme.

---

## ⚙️ Model hyperparameters

Best configuration selected via grid search (paper Table 2):

| Hyperparameter | Value |
|---|---|
| Diffusion steps K | 100 |
| Noise schedule | Linear |
| Base channels | 128 |
| Time embed dim | 32 |
| U-Net depth | 2 |
| Batch size | 256 |
| Epochs | 2000 |
| Learning rate | 1e-3 |
| Conditions | GI, NC, NC lag-1 |

---

## 📄 Citation

If you use this code, please cite the paper:

```bibtex
@article{jene2026cddpm,
  title   = {Conditional diffusion modeling for probabilistic behind-the-meter PV disaggregation},
  author  = {Jené-Vinuesa, Marc and Kazmi, Hussain and Aragüés-Peñalba, Mònica and Sumper, Andreas},
  journal = {Energy and AI},
  year    = {2026},
  doi     = {10.1016/j.egyai.2026.100692}
}
```

---

## 📢 Contact

Marc Jené Vinuesa — marc.jene@upc.edu  
CITCEA-UPC, Universitat Politècnica de Catalunya
