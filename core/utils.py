from pathlib import Path
import pandas as pd
import yaml


"""
Utility functions for saving and loading data.

Functions in this module should handle proper error checking and maintain
backwards compatibility when file formats change.
"""

def save_json(df, path, strftime=True, time_column_name=None):
    if strftime and time_column_name is not None:
        try:
            df[time_column_name] = df[time_column_name].dt.strftime('%Y-%m-%d:%H:%M:%S')
        except AttributeError:
            pass
    df.to_json(path, orient='records', indent=4)

def save_pickle(obj, path):
    """
    Save an object to a pickle file.
    
    Args:
        obj: The object to save.
    """
    with open(path, 'wb') as file:
        pd.to_pickle(obj, file)

def load_pickle(path):
    """
    Load an object from a pickle file.
    """
    with open(path, 'rb') as file:
        return pd.read_pickle(file)

def load_yaml(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

import sys
import subprocess
from datetime import datetime
from pathlib import Path

def save_run_metadata(args, meta_path):
    """Create a log directory and save experiment metadata."""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Customize naming scheme if desired
    exp_name = f"{args.method}_cfg{args.config_id}_exp{args.expid}_{timestamp}"

    log_root = Path("logs")
    log_dir = log_root / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Retrieve git hash
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        # Mark dirty if uncommitted changes exist
        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if dirty != 0:
            git_hash += "-dirty"
    except Exception:
        git_hash = "N/A"

    # Full command
    cmd = " ".join(sys.argv)

    # Save metadata
    with meta_path.open("w") as f:
        f.write(f"date: {now.date()}\n")
        f.write(f"time: {now.time()}\n")
        f.write(f"command: {cmd}\n")
        f.write(f"git_hash: {git_hash}\n\n")

        f.write("=== Args ===\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    print(f"[log] Metadata saved to {meta_path}")
    return log_dir

def create_results_dataframe(alphas, time, horizon, ground_truths_dict, prediction_intervals_dict, scores_dict, qs_dict):
    """
    Construct a DataFrame with one row per (t, alpha, h). Each row has columns:
        - time:                  scalar time[t]
        - alpha:                 scalar alpha value
        - h:                     horizon index (0-based, 0 ≤ h < horizon)
        - ground_truth:          scalar ground_truths_dict[alpha][t, h]
        - prediction_interval:   array of shape (N, 2) for prediction_intervals_dict[alpha][t, h]

    Args:
        alphas:  List (or iterable) of alpha values.
        time:    Numpy array of shape (T,) (shared for all alphas).
        horizon: Integer H (forecast‐horizon length).
        ground_truths_dict: 
                 Dict mapping each alpha → a numpy array of shape (T, H).
        prediction_intervals_dict:
                 Dict mapping each alpha → a numpy array of shape (T, H, N, 2).
    
    Returns:
        pd.DataFrame with columns ["time", "alpha", "h", "ground_truth", "prediction_interval"].
    """
    records = []
    T = len(time)

    for alpha in alphas:
        gt_array = ground_truths_dict[alpha]        # shape: (T, H)
        pi_array = prediction_intervals_dict[alpha] # shape: (T, H, N, 2)
        scores_array = scores_dict[alpha]  # shape: (T, H)
        qs_array = qs_dict[alpha]          # shape: (T, H)
        for t in range(T):
            for h_idx in range(horizon):
                record = {
                    "time": time[t],
                    "alpha": round(alpha, 2),
                    "h": h_idx,
                    "ground_truth": gt_array[t, h_idx],
                    "prediction_interval": pi_array[t, h_idx],  # shape: (N, 2)
                    "scores": scores_array[t, h_idx],
                    "qs": qs_array[t, h_idx]
                }
                records.append(record)
    
    df = pd.DataFrame.from_records(records, columns=[
        "time", "alpha", "h", "ground_truth", "prediction_interval", "scores", "qs"
    ])
    df["alpha"] = df["alpha"].map("{:.2f}".format)
    return df


def create_2dresults_dataframe(alphas, time, horizon, ground_truths_dict, prediction_intervals_dict, scores_dict, qs_dict):
    """    
    Returns:
        pd.DataFrame with columns ["time", "alpha", "h", "ground_truth", "samples", "qs", "scores"].
    """
    records = []
    T = len(time)

    for alpha in alphas:
        gt_array = ground_truths_dict[alpha]        # shape: (T, H)
        sample_array, qs_array = prediction_intervals_dict[alpha] # shape: (T, N, H, 2), (T, H)
        
        for t in range(T):
            for h_idx in range(horizon):
                record = {
                    "time": time[t],
                    "alpha": round(alpha, 2),
                    "h": h_idx,
                    "ground_truth": gt_array[t, h_idx],
                    "samples": sample_array[t, :, h_idx, :],  # shape: (N, 2)
                    "qs": qs_array[t, h_idx],  # scalar value
                    "scores": scores_dict[alpha][t, h_idx]
                }
                records.append(record)
    
    df = pd.DataFrame.from_records(records, columns=[
        "time", "alpha", "h", "ground_truth", "samples", "qs", "scores"
    ])
    df["alpha"] = df["alpha"].map("{:.2f}".format)
    return df