import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from statsmodels.nonparametric.kde import KDEUnivariate

def get_noise_schedule(total_steps: int, schedule_type: str = "linear") -> dict:
    """
    Generate the cumulative product of alphas for the diffusion process.
    :param total_steps: Total number of diffusion steps (T).
    :param schedule_type: Type of noise schedule ("linear", "cosine", etc.).
    :return: Array of the cumulative product of alphas for each timestep t ∈ [0, T-1].
    """
    if schedule_type == "linear":
        noise_start = 1e-4
        noise_end = 0.02
        betas = np.linspace(noise_start, noise_end, total_steps)
    elif schedule_type == "cosine":
        s_offset = 0.008
        steps_range = np.arange(total_steps + 1, dtype=np.float64) / total_steps
        alpha_bar = np.cos((steps_range + s_offset) / (1 + s_offset) * np.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]  # Normalize
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = np.clip(betas, 1e-8, 0.999)
    else:
        raise ValueError("Unsupported noise schedule type.")

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    return {"betas": betas, "alphas": alphas, "alphas_cumprod": alphas_cumprod}


def q_sample(x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """
    Add noise to a clean input x_0 at a given diffusion timestep t.
    :param x_0: Original input tensor.
    :param t: Diffusion timestep.
    :param noise: Noise tensor to be added of same shape as x_0.
    :param alphas_cumprod: Cumulative product of alphas for the diffusion process.
    :return: Noisy input tensor at timestep t.
    """
    if isinstance(t, int):
        t = torch.tensor([t], dtype=torch.long, device=x_0.device)
    else:
        t = t.to(x_0.device)

    sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[t]).unsqueeze(-1)
    sqrt_one_minus = torch.sqrt(1 - alphas_cumprod[t]).unsqueeze(-1)

    # broadcasting across dimensions if needed
    while sqrt_alpha_cumprod.ndim < x_0.ndim:
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
        sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

    return sqrt_alpha_cumprod * x_0 + sqrt_one_minus * noise


def generate_real_pv_ts(pv_tensor: torch.Tensor, *condition_tensors: torch.Tensor, n_samples: int = 64) -> tuple:
    """
    Randomly sample a batch of real PV time series and their corresponding condition signals.

    Args:
        pv_tensor (torch.Tensor): PV time series [N, L]
        *condition_tensors (torch.Tensor): One or more condition tensors [N, L]
        n_samples (int): Number of samples to return.
    Returns:
        tuple: Tuple of (pv_batch, cond1_batch, cond2_batch, ...) of shape [n_samples, L]
    """
    idx = torch.randint(0, len(pv_tensor), (n_samples,))

    sampled = [pv_tensor[idx]] + [cond[idx] for cond in condition_tensors]
    return tuple(sampled)


def reverse_step(x_t, pred_noise, alpha_t, beta_t, alpha_cumprod_t, t) -> torch.Tensor:
    """
    Perform the reverse diffusion step to denoise the input tensor x_t.
    :param x_t: Noisy input tensor at timestep t.
    :param pred_noise: Predicted noise tensor.
    :param alpha_t: alpha_t for the current timestep.
    :param beta_t: beta_t for the current timestep.
    :param alpha_cumprod_t: Cumulative product of alphas for the current timestep.
    :param t: Current timestep.
    :return: Denoised tensor.
    """
    z = torch.randn_like(x_t) if t > 0 else 0
    return (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * pred_noise
                                        ) + torch.sqrt(beta_t) * z

def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_customer(sample_str):
    return int(sample_str.split("_")[0])

def extract_month(sample_str):
    month = int(sample_str.split("_")[1].split("/")[1])
    return month


def split_train_test_tensors(pv_data: pd.DataFrame, cond_data: dict, seed: int, test_size: int, train_size: int,
                              drop_customers: list = None):
    """
    Split PV and condition data into train/test sets and return tensors.
    """
    all_customers = pv_data.index.str.split("_").str[0].astype(int).unique().tolist()
    if drop_customers is not None:
        all_customers = [c for c in all_customers if c not in drop_customers]

    rng = np.random.RandomState(seed)  # NOTE: is this needed?
    customers_test = sorted(rng.choice(all_customers, size=test_size, replace=False))

    customer_ids = pv_data.index.str.split("_").str[0].astype(int)
    test_mask = customer_ids.isin(customers_test)
    if train_size < (len(all_customers) - test_size):
        # Randomly select customers for training
        customers_train = sorted(rng.choice(
            [c for c in all_customers if c not in customers_test], size=train_size, replace=False))
        train_mask = customer_ids.isin(customers_train)
    else:
        train_mask = ~test_mask

    # Convert to tensors
    pv_train = torch.tensor(pv_data[train_mask].values, dtype=torch.float32)
    train_conditions = {}
    for key in cond_data:
        train_conditions[key] = torch.tensor(cond_data[key][train_mask].values, dtype=torch.float32)

    pv_test = torch.tensor(pv_data[test_mask].values, dtype=torch.float32)
    test_index = pv_data[test_mask].index
    test_customers = test_index.str.split("_").str[0].astype(int).tolist()
    test_dates = test_index.str.split("_").str[1].tolist()
    test_conditions = {}
    for key in cond_data:
        test_conditions[key] = torch.tensor(cond_data[key][test_mask].values, dtype=torch.float32)

    return {"pv_train": pv_train, "pv_test": pv_test, "conditions_train": train_conditions,
            "conditions_test": test_conditions, "test_customers": test_customers, "test_dates": test_dates}


def refine_samples(samples: torch.Tensor, gi_cond: torch.Tensor = None, nc_cond: torch.Tensor = None,
                   cap_pred: float = 1.0, max_exp: float = 1.0) -> torch.Tensor:
    """
    Apply rules of thumb to refine generated PV samples.
    :param samples: Generated PV samples tensor of shape (batch_size, 48).
    :param gi_cond: Irradiance condition tensor of shape (48,).
    :param nc_cond: Non-consumption condition tensor of shape (48,).
    :param cap_pred: Predicted capacity for the customer.
    :param max_exp: Maximum export limit for the customer.
    :return: Refined PV samples tensor of shape (batch_size, 48).
    """
    samples_out = samples.detach().cpu()

    if gi_cond is not None:
        # Zero out columns where gi_cond == 0
        gi_mask = (gi_cond != 0).float()  # (48,)
        samples_out = samples_out * gi_mask  # (batch_size, 48)

    if nc_cond is not None:
        scaled_samples = torch.maximum(samples_out*cap_pred, -nc_cond*max_exp)  # (batch_size, 48)
        samples_out = scaled_samples/cap_pred  # Scale back to original capacity

    return samples_out


def compute_kde_peaks(samples_np: np.ndarray) -> np.ndarray:
    """
    Compute the peak (mode) of KDE for each time step with safe handling of edge cases.
    """
    peaks = []
    for i in range(samples_np.shape[1]):
        values = samples_np[:, i]

        # Check for near-zero variance
        if np.allclose(values, values[0], atol=1e-8):
            peaks.append(values[0])
            continue

        kde = KDEUnivariate(values)
        try:
            kde.fit()
        except RuntimeError:
            # Fallback: assume peak is the mean
            peaks.append(np.mean(values))
            continue

        # Evaluate over a dense grid
        grid = np.linspace(values.min(), values.max(), 1000)
        density = kde.evaluate(grid)
        peak = grid[np.argmax(density)]
        peaks.append(peak)

    return np.array(peaks)


def compute_pv_stats(customer_id: str, date: str, samples: torch.Tensor, real: np.ndarray, cap: float = 1.0) -> dict:
    """
    Compute summary statistics over generated samples.
    """
    samples_np = samples.detach().cpu().numpy() * cap
    ts_length = samples_np.shape[1]
    minutes_step = 24*60 // ts_length
    return {
        "customer": np.full(ts_length, customer_id),
        "date": np.full(ts_length, date),
        "hour": ((np.arange(ts_length) * minutes_step) // 60).astype(int),
        "minutes": ((np.arange(ts_length) * minutes_step) % 60).astype(int),
        "real": real,
        "mean": np.mean(samples_np, axis=0),
        "median": np.median(samples_np, axis=0),
        "kde_peak": compute_kde_peaks(samples_np),
        "std": np.std(samples_np, axis=0),
        "p0": np.percentile(samples_np, 0, axis=0),
        "p10": np.percentile(samples_np, 10, axis=0),
        "p20": np.percentile(samples_np, 20, axis=0),
        "p30": np.percentile(samples_np, 30, axis=0),
        "p40": np.percentile(samples_np, 40, axis=0),
        "p50": np.percentile(samples_np, 50, axis=0),
        "p60": np.percentile(samples_np, 60, axis=0),
        "p70": np.percentile(samples_np, 70, axis=0),
        "p80": np.percentile(samples_np, 80, axis=0),
        "p90": np.percentile(samples_np, 90, axis=0),
        "p100": np.percentile(samples_np, 100, axis=0),
        "p25": np.percentile(samples_np, 25, axis=0),
        "p75": np.percentile(samples_np, 75, axis=0),
        "p5": np.percentile(samples_np, 5, axis=0),
        "p95": np.percentile(samples_np, 95, axis=0),
    }

def get_results_customer(stats, seed, time_id, save=True):
    """
    Convert the dictionary of statistics into a DataFrame and save it as a CSV file.
    """
    customer = str(stats["customer"][0][0])
    for key in stats.keys():
        stats[key] = np.concatenate(stats[key], axis=0)
    df_results = pd.DataFrame(stats)

    if save:
        df_results.to_csv("prob_disaggregation/pvdiff_seed" + str(seed) + "_c" + customer + "_" + str(time_id) + ".csv", index=False)
    return df_results


def get_mase(real: np.ndarray, estimation: np.ndarray) -> float:
    """
    Compute Mean Absolute Scaled Error (MASE) for PV disaggregation.
    """
    # Naive forecast obtained with the previous time step
    mae = np.mean(np.abs(real - estimation))
    mae_naive = np.mean(np.abs(real[1:] - real[:-1]))
    if mae_naive == 0:
        raise ValueError("Naive MAE is zero, cannot compute MASE.")
    else:
        return mae / mae_naive


def get_mape(df_results: pd.DataFrame, estimation_column: str) -> float:
    """
    Compute Mean Absolute Percentage Error (MAPE) for PV disaggregation.
    """
    # Avoid division by zero
    real = df_results["real"].values
    estimation = df_results[estimation_column].values
    # Add additional filter to avoid division by zero or very small values
    mask = (real > 0.01)
    mape = np.mean(np.abs((real[mask] - estimation[mask]) / real[mask])) * 100
    return mape


def get_smape(df_results: pd.DataFrame, estimation_column: str) -> float:
    """
    Compute Symmetric Mean Absolute Percentage Error (SMAPE) for PV disaggregation.
    """
    real = df_results["real"].values
    estimation = df_results[estimation_column].values
    # Add additional filter to avoid division by zero or very small values
    mask = (real > 0.01)
    smape = np.mean(2 * np.abs(real[mask] - estimation[mask]) / (np.abs(real[mask]) + np.abs(estimation[mask]))) * 100
    return smape


def get_picp(df_results: pd.DataFrame, ci: int) -> float:
    """
    Compute Prediction Interval Coverage Probability (PICP) for PV disaggregation.
    """
    real = df_results["real"].values
    lb = int(50 - ci/2)
    ub = 100 - lb
    lower_bound = df_results[f"p{lb}"].values
    upper_bound = df_results[f"p{ub}"].values

    # Count the number of times the real value is within the prediction interval
    count_within_interval = np.sum((real >= lower_bound) & (real <= upper_bound))
    picp = count_within_interval / len(real) * 100
    return picp


def get_pinaw(df_results: pd.DataFrame, capacity, ci: int) -> float:
    """
    Compute Prediction Interval Normalized Width (PINW) for PV disaggregation.
    """
    lb = int(50 - ci/2)
    ub = 100 - lb
    lower_bound = df_results[f"p{lb}"].values
    upper_bound = df_results[f"p{ub}"].values

    # Calculate the width of the prediction interval
    pinaw = np.mean(upper_bound - lower_bound)/capacity
    return pinaw


def crps_from_quantiles(df_results):
    """
    Compute Continuous Ranked Probability Score (CRPS) from quantiles.
    """
    y_true = df_results["real"].values
    q_columns = ["p" + str(int(q)) for q in np.arange(0, 101, 10)]
    quantiles = df_results[q_columns].values
    q_values = np.arange(0, 101, 10) / 100.0

    N, Q = quantiles.shape
    crps = np.zeros(N)

    for i in range(N):
        F = q_values
        z = quantiles[i]
        y = y_true[i]

        if np.allclose(z, z[0]):
            # All percentiles equal → deterministic forecast → CRPS = MAE
            crps[i] = abs(z[0] - y)
        else:
            indicator = np.asarray(z >= y).astype(float)
            diff = F - indicator
            dz = np.gradient(z)
            crps[i] = np.sum((diff ** 2) * dz)

    crps_mean = np.mean(crps)
    return crps_mean


def calculate_pinball_loss(df_results, quantiles=np.arange(0.1, 1.0, 0.1)):
    """
    Calculate the average pinball loss for the given quantiles.
    """
    y_true = df_results['real'].values

    pl_total = 0
    valid_quantiles = 0

    for q in quantiles:
        column = f"p{int(q * 100)}"
        if column not in df_results.columns:
            continue  # Skip missing quantiles
        q_pred = df_results[column].values
        errors = y_true - q_pred
        loss = np.where(errors >= 0, q * errors, (1 - q) * (-errors))
        pl_total += np.mean(loss)
        valid_quantiles += 1

    if valid_quantiles == 0:
        raise ValueError("No valid quantile columns found in df_results")

    avg_pinball_loss = pl_total / valid_quantiles
    return avg_pinball_loss



def get_metrics(df_results: pd.DataFrame, capacity: float, deterministic_values: list) -> dict:
    """
    Compute basic evaluation metrics between predicted stats and real PV.
    """
    dict_output = {}
    for estimation_column in deterministic_values:
        real = df_results["real"].values
        estimation = df_results[estimation_column].values
        customer = df_results["customer"].values[0]

        # replace NaNs in estimation for 0
        estimation = np.nan_to_num(estimation, nan=0.0)

        # Deterministic metrics
        df_deterministic = df_results[["real", estimation_column, "hour"]].copy()
        rmse = np.sqrt(np.mean((real - estimation) ** 2))
        mae = np.mean(np.abs(real - estimation))
        mase = get_mase(real, estimation)
        r2 = r2_score(real, estimation)
        # Filter between 10h and 16h for MAPE and SMAPE
        mask = (df_deterministic["hour"] >= 10) & (df_deterministic["hour"] <= 15)
        df_det_filtered = df_deterministic[mask]
        mape = get_mape(df_det_filtered, estimation_column)
        smape = get_smape(df_det_filtered, estimation_column)

        # Store results in dictionary
        dict_output["rmse_"+estimation_column] = rmse
        dict_output["mae_"+estimation_column] = mae
        dict_output["mase_"+estimation_column] = mase
        dict_output["r2_"+estimation_column] = r2
        dict_output["mape_"+estimation_column] = mape
        dict_output["smape_"+estimation_column] = smape

    # Probabilistic metrics
    mask_picp = (df_results["hour"] >= 5) & (df_results["hour"] < 21)
    df_prob_filtered = df_results[mask_picp]
    # Prediction Interval Coverage Probability (PICP)
    picp_50 = get_picp(df_prob_filtered, 50)
    picp_80 = get_picp(df_prob_filtered, 80)
    picp_90 = get_picp(df_prob_filtered, 90)
    # Prediction Interval Width (PIW)
    pinaw_50 = get_pinaw(df_prob_filtered, capacity, 50)
    pinaw_80 = get_pinaw(df_prob_filtered, capacity, 80)
    pinaw_90 = get_pinaw(df_prob_filtered, capacity, 90)
    # Continuous Ranked Probability Score (CRPS)
    crps_all = crps_from_quantiles(df_results)
    crps_sun = crps_from_quantiles(df_prob_filtered)
    # Pinball loss
    pinball_all = calculate_pinball_loss(df_results)
    pinball_sun = calculate_pinball_loss(df_prob_filtered)

    # Store probabilistic metrics
    dict_output["picp_50"] = picp_50
    dict_output["picp_80"] = picp_80
    dict_output["picp_90"] = picp_90
    dict_output["pinaw_50"] = pinaw_50
    dict_output["pinaw_80"] = pinaw_80
    dict_output["pinaw_90"] = pinaw_90
    dict_output["crps_all"] = crps_all
    dict_output["crps_sun"] = crps_sun
    dict_output["pl_all"] = pinball_all
    dict_output["pl_sun"] = pinball_sun
    dict_output["customer"] = customer

    return dict_output


def metrics_sensitivity(df_results: pd.DataFrame, capacity: float, deterministic_values: str, sensitivity: str):
    """
    Compute sensitivity metrics for the given deterministic values.
    :param df_results: DataFrame with results.
    :param capacity: Capacity of the PV system.
    :param deterministic_values: List of deterministic value columns to analyze.
    :param sensitivity: Type of sensitivity analysis to perform ('hour' or 'season').
    :return: Dictionary with sensitivity metrics.
    """
    dict_output = {}

    if sensitivity not in ['hour', 'season']:
        raise ValueError("Sensitivity must be one of ['hour', 'season']")

    if deterministic_values not in ['mean', 'median', 'kde_peak']:
        raise ValueError("Deterministic values must be one of ['mean', 'median', 'kde_peak']")

    filtered_df = df_results.copy()
    df_output = pd.DataFrame()

    if sensitivity == 'hour':
        for h in range(6, 20, 2):
            mask = (filtered_df["hour"] >= h) & (df_results["hour"] < h+2)
            df_h = filtered_df[mask]
            real_h = df_h["real"].values
            estimation_h = df_h[deterministic_values].values

            # Deterministic metrics
            rmse = np.sqrt(np.mean((real_h - estimation_h) ** 2))
            mae = np.mean(np.abs(real_h - estimation_h))
            mase = get_mase(real_h, estimation_h)
            r2 = r2_score(real_h, estimation_h)
            mape = get_mape(df_h, deterministic_values)
            smape = get_smape(df_h, deterministic_values)

            # Probabilistic metrics
            picp_50 = get_picp(df_h, 50)
            picp_80 = get_picp(df_h, 80)
            picp_90 = get_picp(df_h, 90)
            pinaw_50 = get_pinaw(df_h, capacity, 50)
            pinaw_80 = get_pinaw(df_h, capacity, 80)
            pinaw_90 = get_pinaw(df_h, capacity, 90)
            crps_sun = crps_from_quantiles(df_h)
            pinball_sun = calculate_pinball_loss(df_h)
            # Store results in dictionary
            metrics_h = {'hour': str(h)+'-'+str(h+2), 'customer': df_h["customer"].values[0],
                         'rmse': rmse, 'mae': mae, 'mase': mase, 'r2': r2,
                         'mape': mape, 'smape': smape,
                         'picp_50': picp_50, 'picp_80': picp_80, 'picp_90': picp_90,
                         'pinaw_50': pinaw_50, 'pinaw_80': pinaw_80, 'pinaw_90': pinaw_90,
                         'crps_sun': crps_sun, 'pl_sun': pinball_sun}
            df_output = pd.concat([df_output, pd.DataFrame(metrics_h, index=[0])], ignore_index=True)

    elif sensitivity == 'season':
        filtered_df["month"] = filtered_df["date"].str[3:5].astype(int)
        for month in [6, 9, 12, 3]:
            if month == 12:
                mask = (filtered_df["month"] >= month) | (filtered_df["month"] < 3)
            else:
                mask = (filtered_df["month"] >= month) & (filtered_df["month"] < month + 3)
            df_m = filtered_df[mask]
            real_m = df_m["real"].values
            estimation_m = df_m[deterministic_values].values

            # Deterministic metrics
            rmse = np.sqrt(np.mean((real_m - estimation_m) ** 2))
            mae = np.mean(np.abs(real_m - estimation_m))
            mase = get_mase(real_m, estimation_m)
            r2 = r2_score(real_m, estimation_m)
            mask_mape = (df_m["hour"] >= 10) & (df_m["hour"] <= 15)
            df_m_mape = df_m[mask_mape]
            mape = get_mape(df_m_mape, deterministic_values)
            smape = get_smape(df_m_mape, deterministic_values)

            # Probabilistic metrics
            mask_picp = (df_m["hour"] >= 5) & (df_m["hour"] < 21)
            df_m_picp = df_m[mask_picp]
            picp_50 = get_picp(df_m_picp, 50)
            picp_80 = get_picp(df_m_picp, 80)
            picp_90 = get_picp(df_m_picp, 90)
            pinaw_50 = get_pinaw(df_m_picp, capacity, 50)
            pinaw_80 = get_pinaw(df_m_picp, capacity, 80)
            pinaw_90 = get_pinaw(df_m_picp, capacity, 90)
            crps_sun = crps_from_quantiles(df_m_picp)
            pinball_sun = calculate_pinball_loss(df_m_picp)

            # Store results in dictionary
            metrics_month = {'starting_month': str(month), 'customer': df_m["customer"].values[0],
                             'rmse': rmse, 'mae': mae, 'mase': mase, 'r2': r2,
                             'mape': mape, 'smape': smape,
                             'picp_50': picp_50, 'picp_80': picp_80, 'picp_90': picp_90,
                             'pinaw_50': pinaw_50, 'pinaw_80': pinaw_80, 'pinaw_90': pinaw_90,
                             'crps_sun': crps_sun, 'pl_sun': pinball_sun}
            df_output = pd.concat([df_output, pd.DataFrame(metrics_month, index=[0])], ignore_index=True)

    return df_output



def plot_pv_uncertainty(real: torch.Tensor, stats: dict, gi: torch.Tensor = None):
    """
    Plot PV prediction confidence intervals and compare to real PV.
    """
    x = np.arange(len(real))
    real_np = real.detach().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(x, real_np, label="Real PV", color="black")
    plt.plot(x, stats["median"], label="Median", color="blue")
    plt.fill_between(x, stats["p25"], stats["p75"], alpha=0.3, color="blue", label="50% CI")
    plt.fill_between(x, stats["p10"], stats["p90"], alpha=0.2, color="blue", label="80% CI")

    if gi is not None:
        plt.plot(x, gi.detach().cpu().numpy(), linestyle="--", label="Irradiance", color="orange")

    plt.ylim(0, 1)
    plt.legend()
    plt.title("PV Prediction with Uncertainty")
    plt.tight_layout()
    plt.show()


def plot_training_loss(losses, run_id):
    plt.figure()
    plt.plot(losses)
    # plot rolling mean with window size 10
    plt.plot(pd.Series(losses).rolling(window=10).mean(), color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.savefig(f"figures/loss_curve_{run_id}.png")
    plt.close()
