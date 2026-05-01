"""
main.py — Unified training and evaluation script for probabilistic BTM PV disaggregation.

Supports three models selectable via the 'model' key in the config dict:
  cddpm  Conditional Denoising Diffusion Probabilistic Model (paper main model, Table 2)
  cvae   Conditional Variational Autoencoder (benchmark)
  qr     Quantile Regression with 1D CNN (benchmark)

Command-line usage:
    python main.py                    # cDDPM with best config, seed 0
    python main.py --model cvae
    python main.py --model qr
    python main.py --model cddpm --seed 2

Programmatic usage (e.g. from a notebook):
    from main import run, BEST_CONFIGS
    df_metrics = run(BEST_CONFIGS["cddpm"])
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils import (
    set_seed,
    split_train_test_tensors,
    refine_samples,
    compute_pv_stats,
    get_results_customer,
    get_metrics,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Customers excluded due to anomalous or incomplete data (see paper Section 3.1)
# ---------------------------------------------------------------------------

DROP_CUSTOMERS = [
    1, 2, 3, 6, 7, 8, 9, 11, 22, 24, 26, 29, 41, 57, 61, 79, 91, 92, 94, 97,
    103, 115, 116, 135, 138, 146, 150, 158, 187, 190, 203, 229, 231, 234, 245,
    249, 250, 252, 265, 266, 275, 286, 300,
]

# ---------------------------------------------------------------------------
# Best configurations
# ---------------------------------------------------------------------------

BEST_CONFIGS: dict = {
    # cDDPM — selected hyperparameters from Table 2 of the paper.
    # Each config was evaluated over 10 seeds; results were averaged.
    "cddpm": {
        "model": "cddpm",
        "seed": 0,
        "test_size": 150,
        "train_size": 107,
        "conditions": ["gi", "nc", "nc_lag1"],
        "n_samples": 50,
        "ratio_test": 0.2,          # 20% of test days, stratified by season
        "ts_length": 48,            # 30-min resolution -> 48 slots per day
        # Architecture (Table 2)
        "diffusion_steps": 100,
        "noise_schedule": "linear",
        "base_channels": 128,
        "time_embed_dim": 32,
        "unet_depth": 2,
        "n_epochs": 2000,
        "batch_size": 256,
        "learning_rate": 1e-3,
    },
    # cVAE — conditional VAE benchmark.
    "cvae": {
        "model": "cvae",
        "seed": 0,
        "test_size": 150,
        "train_size": 107,
        "conditions": ["gi", "nc", "nc_lag1"],
        "n_samples": 50,
        "ratio_test": 0.2,
        "ts_length": 48,
        "z_dim": 16,
        "z_temp": 2.0,              # latent temperature (diversity control)
        "hidden": 256,
        "n_epochs": 2000,
        "batch_size": 256,
        "learning_rate": 5e-3,
        "beta_max": 0.01,
        "warmup_frac": 0.7,
    },
    # QR — Quantile Regression with 1D CNN benchmark.
    # Predicts Q=21 quantile curves passed directly to the evaluation pipeline.
    "qr": {
        "model": "qr",
        "seed": 0,
        "test_size": 150,
        "train_size": 107,
        "conditions": ["gi", "nc", "nc_lag1"],
        "ratio_test": 0.2,
        "ts_length": 48,
        "n_epochs": 1000,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "qr_base": 64,
    },
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(config: dict) -> tuple:
    """Load and split the Ausgrid dataset."""
    def read(fname):
        return pd.read_csv(DATA_DIR / fname, index_col=0)

    pv_data       = read("pv_norm_cap_v2.csv")
    gi_data       = read("ghi_norm_v2.csv")
    nc_data       = read("nc_norm_maxexp_v2.csv")
    nc_lag1_data  = read("nc_lag1_maxexp_v2.csv")
    capacity_data = read("capacity_estimation_GG_pred.csv")

    all_conditions = {"gi": gi_data, "nc": nc_data, "nc_lag1": nc_lag1_data}
    cond_data = {k: v for k, v in all_conditions.items() if k in config["conditions"]}

    exp_data = split_train_test_tensors(
        pv_data, cond_data,
        seed=config["seed"],
        test_size=config["test_size"],
        train_size=config["train_size"],
        drop_customers=config.get("drop_customers", DROP_CUSTOMERS),
    )
    return exp_data, capacity_data


# ---------------------------------------------------------------------------
# Model dispatch — training
# ---------------------------------------------------------------------------

def _train(config: dict, exp_data: dict):
    """Train the selected model and return it."""
    model_type = config["model"]
    device = config["device"]

    if model_type == "cddpm":
        from core import train_model
        model, _ = train_model(
            pv_train=exp_data["pv_train"],
            conditions_train=exp_data["conditions_train"],
            diffusion_steps=config["diffusion_steps"],
            in_ch=len(config["conditions"]) + 1,
            base_channels=config["base_channels"],
            time_embed_dim=config["time_embed_dim"],
            n_epochs=config["n_epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            noise_schedule=config["noise_schedule"],
            unet_depth=config["unet_depth"],
            device=device,
        )

    elif model_type == "cvae":
        from models.cvae import train_cvae, build_features_flat
        cond_flat = build_features_flat(exp_data["conditions_train"], config["conditions"])
        model = train_cvae(
            pv_train=exp_data["pv_train"],
            cond_train_flat=cond_flat,
            pv_dim=config["ts_length"],
            z_dim=config["z_dim"],
            hidden=config["hidden"],
            n_epochs=config["n_epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            beta_max=config["beta_max"],
            warmup_frac=config["warmup_frac"],
            device=device,
        )

    elif model_type == "qr":
        from models.quantile import train_quantile_regressor
        model = train_quantile_regressor(
            pv_train=exp_data["pv_train"],
            cond_train=exp_data["conditions_train"],
            cond_order=config["conditions"],
            n_epochs=config["n_epochs"],
            batch_size=config["batch_size"],
            lr=config["learning_rate"],
            base=config["qr_base"],
            device=device,
        )

    else:
        raise ValueError(f"Unknown model '{model_type}'. Choose from: cddpm, cvae, qr")

    return model


# ---------------------------------------------------------------------------
# Model dispatch — inference
# ---------------------------------------------------------------------------

def _generate(model, conditions_test: dict, config: dict) -> torch.Tensor:
    """
    Generate probabilistic PV samples for a single day.
    Returns [S, T] where S = n_samples (cddpm/cvae) or Q=21 quantiles (qr).
    """
    model_type = config["model"]
    device = config["device"]

    if model_type == "cddpm":
        from core import sample
        return sample(
            model=model,
            conditions_dict=conditions_test,
            n_samples=config["n_samples"],
            ts_length=config["ts_length"],
            diffusion_steps=config["diffusion_steps"],
            noise_schedule=config["noise_schedule"],
            device=device,
        )

    elif model_type == "cvae":
        from models.cvae import sample_cvae, build_features_flat
        cond_flat = build_features_flat(
            {k: v.unsqueeze(0) for k, v in conditions_test.items()},
            config["conditions"],
        ).squeeze(0)
        return sample_cvae(
            model=model,
            cond_one_flat=cond_flat,
            n_samples=config["n_samples"],
            z_temp=config.get("z_temp", 2.0),
            device=device,
        )

    elif model_type == "qr":
        from models.quantile import predict_quantiles
        return predict_quantiles(
            model=model,
            cond_one=conditions_test,
            cond_order=config["conditions"],
            device=device,
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(config: dict) -> pd.DataFrame:
    """
    Full train-evaluate pipeline for a single configuration.

    Args:
        config: Dict with model hyperparameters (see BEST_CONFIGS for reference).

    Returns:
        DataFrame with per-customer evaluation metrics.
    """
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "drop_customers": DROP_CUSTOMERS,
        **config,
    }

    print(f"\n{'='*60}")
    print(f"  Model : {config['model'].upper()}")
    print(f"  Seed  : {config['seed']}")
    print(f"  Device: {config['device']}")
    print(f"{'='*60}\n")

    set_seed(config["seed"])

    # Load data
    exp_data, capacity_data = load_data(config)

    # Train
    t0 = time.time()
    model = _train(config, exp_data)
    print(f"\nTraining done in {time.time() - t0:.1f}s\n")

    # Seasonal stratified test sampling
    dates = pd.to_datetime(exp_data["test_dates"], format="%d/%m/%Y")
    season_groups = defaultdict(lambda: defaultdict(list))
    for idx, (cust, date) in enumerate(zip(exp_data["test_customers"], dates)):
        season_groups[cust][(date.month - 1) // 3].append(idx)

    ratio_test = config.get("ratio_test", 1.0)
    if ratio_test == 1:
        selected_indices = list(range(len(exp_data["pv_test"])))
    else:
        rng = np.random.default_rng(seed=config["seed"])
        selected_indices = []
        for cust, seasons in season_groups.items():
            for season, indices in seasons.items():
                n = max(1, int(len(indices) * ratio_test))
                selected_indices.extend(rng.choice(indices, size=n, replace=False).tolist())

    # Evaluate
    time_id = int(time.time())
    prev_customer = exp_data["test_customers"][selected_indices[0]]
    continuos_stats = {}
    all_metrics = []
    cap_real = 0.0
    deterministic_values = ["mean", "median", "kde_peak"]

    for sample_idx in selected_indices:
        real_pv = exp_data["pv_test"][sample_idx]
        conditions_test = {k: v[sample_idx] for k, v in exp_data["conditions_test"].items()}
        customer_id = exp_data["test_customers"][sample_idx]
        date = exp_data["test_dates"][sample_idx]

        if customer_id != prev_customer:
            df_results = get_results_customer(continuos_stats, config["seed"], time_id, save=False)
            all_metrics.append(get_metrics(df_results, cap_real, deterministic_values))
            prev_customer = customer_id
            continuos_stats = {}

        generated = _generate(model, conditions_test, config)

        cap_customer = capacity_data.loc[customer_id, "cap_pred"]
        cap_real     = capacity_data.loc[customer_id, "cap_real"]
        max_exp      = capacity_data.loc[customer_id, "max_export"]
        gi_cond = conditions_test.get("gi") if "gi" in config["conditions"] else None
        refined = refine_samples(generated, gi_cond, None, cap_customer, max_exp)

        real_pv_kw = real_pv * cap_real
        stats = compute_pv_stats(customer_id, date, refined, real_pv_kw.numpy(), cap=cap_customer)
        for k, v in stats.items():
            continuos_stats.setdefault(k, []).append(v)

    # Flush last customer
    df_results = get_results_customer(continuos_stats, config["seed"], time_id, save=False)
    all_metrics.append(get_metrics(df_results, cap_real, deterministic_values))

    df_metrics = pd.DataFrame(all_metrics)
    print(f"\nAverage metrics across {len(df_metrics)} customers:")
    print(df_metrics.mean(numeric_only=True).to_string())
    return df_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate probabilistic PV disaggregation models."
    )
    parser.add_argument(
        "--model", type=str, default="cddpm", choices=["cddpm", "cvae", "qr"],
        help="Model to run (default: cddpm)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default: 0)"
    )
    args = parser.parse_args()

    run({**BEST_CONFIGS[args.model], "seed": args.seed})
