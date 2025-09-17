from collections import defaultdict
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import time

from core import train_model, sample
from utils import *

best_config = {
        "seed": 2,
        "test_size": 5,
        "diffusion_steps": 200,
        "noise_schedule": "linear",
        "base_channels": 64,
        "time_embed_dim": 16,
        "n_epochs": 2000,         # reduce for quick testing
        "batch_size": 64,
        "learning_rate": 1e-3,
        "n_samples": 50,
        "conditions": ["gi", "nc", "nc_lag1"],  # conditions to use
        "drop_customers": [1, 2, 3, 6, 7, 8, 9, 11, 22, 24, 26, 29, 41, 57, 61, 79, 91, 92, 94, 97, 103, 115, 116, 135,
                           138, 146, 150, 158, 187, 190, 203, 229, 231, 234, 245, 249, 250, 252, 265, 266, 275, 286,
                           300],
        "ts_length": 48,  # THIS PARAMETER DEPENDS ON THE DATASET, MAYBE WE SHOULDN"T INCLUDE IT IN THE CONFIG
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "message": "NC normalized using maxexp, PV with predicted capacity"
    }

# Optional: define this inside utils_der.py if you haven’t yet
def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_customer(sample_str):
    return int(sample_str.split("_")[0])

def main(config):
    # Obtain ID
    time_id = int(time.time())

    set_seed(config["seed"])

    # 2. Load and preprocess data
    #### FOR UPC CLUSTER
    if config["device"] == "cuda":
        pv_data = pd.read_csv("code_reproducibility/input_data/pv_norm_cap_v2.csv", index_col=0)
        gi_data = pd.read_csv("code_reproducibility/input_data/ghi_norm_v2.csv", index_col=0)
        cc_data = pd.read_csv("code_reproducibility/input_data/cc_norm_v2.csv", index_col=0)
        nc_data = pd.read_csv("code_reproducibility/input_data/nc_norm_maxexp_v2.csv", index_col=0)
        nc_lag1_data = pd.read_csv("code_reproducibility/input_data/nc_lag1_maxexp_v2.csv", index_col=0)
        capacity_data = pd.read_csv("code_reproducibility/input_data/capacity_estimation_GG_pred.csv", index_col=0)

    #### FOR LOCAL TESTING
    else:
        pv_data = pd.read_csv("input_data/pv_norm_cap_v2.csv", index_col=0)
        gi_data = pd.read_csv("input_data/ghi_norm_v2.csv", index_col=0)
        cc_data = pd.read_csv("input_data/cc_norm_v2.csv", index_col=0)
        nc_data = pd.read_csv("input_data/nc_norm_maxexp_v2.csv", index_col=0)
        nc_lag1_data = pd.read_csv("input_data/nc_lag1_maxexp_v2.csv", index_col=0)
        capacity_data = pd.read_csv("input_data/capacity_estimation_GG_pred.csv", index_col=0)

    conditions_all = {"gi": gi_data, "cc": cc_data, "nc": nc_data, "nc_lag1": nc_lag1_data}
    # Filter based on config conditions
    condition_data = {key: val for key, val in conditions_all.items() if key in config["conditions"]}
    exp_data = split_train_test_tensors(pv_data, condition_data, config["seed"],
                                        config["test_size"], 150, drop_customers=config["drop_customers"])

    # 3. Train model
    model, losses = train_model(
        pv_train=exp_data["pv_train"],
        conditions_train=exp_data["conditions_train"],
        diffusion_steps=config["diffusion_steps"],
        in_ch=len(config["conditions"])+1,
        base_channels=config["base_channels"],
        time_embed_dim=config["time_embed_dim"],
        n_epochs=config["n_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        noise_schedule=config["noise_schedule"],
        device=config["device"]
    )

    #### NEED TO CREATE FUNCTION TO GENERATE SAMPLES FOR ONE OR MORE DAYS ######
    prev_customer = exp_data["test_customers"][0]
    continuos_stats = {}
    all_metrics = []
    for sample_idx in range(0, len(exp_data["pv_test"])):
        real_pv = exp_data["pv_test"][sample_idx]
        conditions_test = {key: val[sample_idx] for key, val in exp_data["conditions_test"].items()}
        customer_id = exp_data["test_customers"][sample_idx]
        date = exp_data["test_dates"][sample_idx]
        if customer_id != prev_customer:
            print("Customer disaggregated:", prev_customer)
            print("Calculating metrics...")
            df_results = get_results_customer(continuos_stats, config["seed"], time_id)
            metrics = get_metrics(df_results, "median")
            all_metrics.append(metrics)
            prev_customer = customer_id
            continuos_stats = {}
            print("NEW CUSTOMER!")
        print("Customer:", customer_id)
        print("Date:", date)

        # 4. Generate samples
        generated = sample(
            model=model,
            conditions_dict=conditions_test,
            n_samples=config["n_samples"],
            ts_length=config["ts_length"],
            diffusion_steps=config["diffusion_steps"],
            noise_schedule=config["noise_schedule"],
            device=config["device"]
        )
        # 4.1. Add rules of thumb based on irradiance and net consumption
        refined_samples = refine_samples(generated, conditions_test["gi"])

        # 5. Compute stats and metrics
        cap_customer = capacity_data.loc[customer_id, "cap_pred"]
        cap_real = capacity_data.loc[customer_id, "cap_real"]
        real_pv = real_pv * cap_real
        stats = compute_pv_stats(customer_id, date, refined_samples, real_pv, cap=cap_customer)

        # Append daily curves
        for key in stats.keys():
            if key in continuos_stats:
                continuos_stats[key].append(stats[key])
            else:
                continuos_stats[key] = [stats[key]]

    df_results = get_results_customer(continuos_stats, config["seed"], time_id)
    metrics = get_metrics(df_results, "median")
    all_metrics.append(metrics)
    df_metrics = pd.DataFrame(all_metrics)
    print("Metrics:", df_metrics)

    # Save metrics and config to Excel
    file_name = "article/metrics_" + str(config["seed"]) + "_" + str(time_id) + ".xlsx"
    config_df = pd.DataFrame(list(config.items()), columns=["parameter", "value"])
    with pd.ExcelWriter(file_name) as writer:
        df_metrics.to_excel(writer, index=False, sheet_name="Metrics")
        config_df.to_excel(writer, index=False, sheet_name="Config")


if __name__ == "__main__":
    main(best_config)
