# matplotlib.use("TkAgg")
import copy
import itertools
import os
import time
from collections import defaultdict

from core import train_model, sample
from utils import *

# 1. Configuration
default_config = {
    "seed": 0,
    "test_size": 150,
    "train_size": 107,  # Maximum possible size for training set after dropping irregular customers
    "diffusion_steps": 200,
    "noise_schedule": "linear",
    "base_channels": 64,
    "time_embed_dim": 16,
    "n_epochs": 2000,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "unet_depth": 2,
    "n_samples": 50,
    "conditions": ["gi", "nc", "nc_lag1"],
    "drop_customers": [1, 2, 3, 6, 7, 8, 9, 11, 22, 24, 26, 29, 41, 57, 61, 79, 91, 92, 94, 97, 103, 115, 116, 135,
                       138, 146, 150, 158, 187, 190, 203, 229, 231, 234, 245, 249, 250, 252, 265, 266, 275, 286,
                       300],
    "ts_length": 48,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "message": "-",
    "config_id": None,  # set later in the loop
    "ratio_test": 0.2,  # ratio of test samples per customer
}

def main(config):
    # Obtain ID
    time_id = int(time.time())
    set_seed(config["seed"])

    # Load and preprocess data
    #### FOR UPC CLUSTER
    if config["device"] == "cuda":
        # Load data from the cluster path
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
    exp_data = split_train_test_tensors(pv_data, condition_data, config["seed"], config["test_size"],
                                        config["train_size"], drop_customers=config["drop_customers"])

    # Train model
    time_before = time.time()
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
        unet_depth= config["unet_depth"],
        device=config["device"]
    )
    time_training = time.time() - time_before

    # plot_training_loss(losses, run_id=str(config["seed"])+"_"+str(config["config_id"])+"_"+str(time_id))

    # Group indices by customer and season
    season_groups = defaultdict(lambda: defaultdict(list))
    dates = pd.to_datetime(exp_data["test_dates"], format="%d/%m/%Y")

    for idx, (cust, date) in enumerate(zip(exp_data["test_customers"], dates)):
        season = (date.month - 1) // 3  # 0:Winter, 1:Spring, 2:Summer, 3:Fall
        season_groups[cust][season].append(idx)

    # Sample % from each seasonal group per customer
    ratio_test = config["ratio_test"]
    if ratio_test == 1:
        selected_indices = list(range(len(exp_data["pv_test"])))
    else:
        selected_indices = []
        rng = np.random.default_rng(seed=config["seed"])

        for cust, seasons in season_groups.items():
            for season, indices in seasons.items():
                n = max(1, int(len(indices) * ratio_test))
                selected = rng.choice(indices, size=n, replace=False)
                selected_indices.extend(selected.tolist())

    prev_customer = exp_data["test_customers"][selected_indices[0]]
    continuos_stats = {}
    all_metrics = []
    cap_real = 0
    deterministic_values = ["mean", "median", "kde_peak"]

    for sample_idx in selected_indices:
        real_pv = exp_data["pv_test"][sample_idx]
        conditions_test = {key: val[sample_idx] for key, val in exp_data["conditions_test"].items()}
        customer_id = exp_data["test_customers"][sample_idx]
        date = exp_data["test_dates"][sample_idx]

        if customer_id != prev_customer:
            print("Customer disaggregated:", prev_customer)
            print("Calculating metrics...")
            if config["seed"] == 2:
                # Save the results for this seed
                df_results = get_results_customer(continuos_stats, config["seed"], time_id, save=True)
            else:
                df_results = get_results_customer(continuos_stats, config["seed"], time_id, save=False)
            metrics = get_metrics(df_results, cap_real, deterministic_values)
            all_metrics.append(metrics)
            prev_customer = customer_id
            continuos_stats = {}
            print("NEW CUSTOMER!")

        print("Customer:", customer_id)
        print("Date:", date)

        # Generate samples
        generated = sample(
            model=model,
            conditions_dict=conditions_test,
            n_samples=config["n_samples"],
            ts_length=config["ts_length"],
            diffusion_steps=config["diffusion_steps"],
            noise_schedule=config["noise_schedule"],
            device=config["device"]
        )
        # Add rules of thumb based on irradiance and net consumption
        cap_customer = capacity_data.loc[customer_id, "cap_pred"]
        cap_real = capacity_data.loc[customer_id, "cap_real"]
        max_exp = capacity_data.loc[customer_id, "max_export"]
        condition_gi, condition_nc = None, None
        if "gi" in config["conditions"]:
            condition_gi = conditions_test["gi"]
        # if "nc" in config["conditions"]:
        #     condition_nc = conditions_test["nc"]
        refined_samples = refine_samples(generated, condition_gi, condition_nc, cap_customer, max_exp)

        # Compute stats and metrics
        real_pv = real_pv * cap_real
        stats = compute_pv_stats(customer_id, date, refined_samples, real_pv, cap=cap_customer)

        # Append daily curves
        for key in stats.keys():
            if key in continuos_stats:
                continuos_stats[key].append(stats[key])
            else:
                continuos_stats[key] = [stats[key]]

    if config["seed"] == 2:
        # Save the results for this seed
        df_results = get_results_customer(continuos_stats, config["seed"], time_id, save=True)
    else:
        df_results = get_results_customer(continuos_stats, config["seed"], time_id, save=False)
    metrics = get_metrics(df_results, cap_real, deterministic_values)
    all_metrics.append(metrics)
    df_metrics = pd.DataFrame(all_metrics)
    print("Metrics:", df_metrics)

    return df_metrics, config, time_training


if __name__ == "__main__":
    # Define parameters to search
    param_grid = {
        "diffusion_steps": [100],
        "noise_schedule": ["linear"],
        "learning_rate": [1e-3],
        "base_channels": [128],
        "n_epochs": [2000],
        "unet_depth": [2],
        "time_embed_dim": [32],
        "conditions": [["gi", "nc", "nc_lag1"]],
        "batch_size": [256],
        "train_size": [107],
        "test_size": [150],
        "seed": [0, 1, 2],
        "ratio_test": [1],
        "message": ["Best parameters - 3 seeds"],
    }
    keys, values = zip(*param_grid.items())

    summary_rows = []
    all_user_metrics = []

    os.makedirs("grid_results", exist_ok=True)

    for i, v in enumerate(itertools.product(*values)):
        config = copy.deepcopy(default_config)
        config.update(dict(zip(keys, v)))
        config_id = i + 1
        config["config_id"] = config_id
        print(f"\n▶▶ Running configuration {config_id}: {config}")

        start_time = time.time()
        df_metrics, config_used, time_training = main(config)
        run_time = time.time() - start_time

        # Compute average metrics across users
        avg_metrics = df_metrics.mean(numeric_only=True).to_dict()
        avg_metrics["config_id"] = config_id
        avg_metrics["run_time_sec"] = run_time
        avg_metrics["time_training_sec"] = time_training
        avg_metrics.update(config_used)
        summary_rows.append(avg_metrics)

        # Add config_id to individual metrics
        df_metrics["config_id"] = config_id
        all_user_metrics.append(df_metrics)

    # Convert to DataFrames
    df_summary = pd.DataFrame(summary_rows)
    df_all_metrics = pd.concat(all_user_metrics, ignore_index=True)

    # Save to Excel with two sheets
    now = time.strftime("%Y%m%d%H%M%S")
    with pd.ExcelWriter("grid_results/gridsearch_results"+f"_{now}.xlsx") as writer:
        df_summary.to_excel(writer, index=False, sheet_name="Summary")
        df_all_metrics.to_excel(writer, index=False, sheet_name="All_User_Metrics")
