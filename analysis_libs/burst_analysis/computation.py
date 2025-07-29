import pandas as pd
from pathlib import Path
import numpy as np
from burst_analysis.plotting import plot_raster
from burst_analysis.plotting import plot_population_rate
from burst_analysis.detection import BurstDetection

def update_parkinsons_exp_metrics(trains_by_condition, current_df=None):
    new_data = pd.DataFrame([
        {"Sample": cond, "Active Units": len(trains)}
        for cond, trains in trains_by_condition.items()
    ])

    if current_df is None:
        return new_data

    missing_cols = [col for col in new_data.columns if col not in current_df.columns]
    for col in missing_cols:
        current_df[col] = None

    missing_cols = [col for col in current_df.columns if col not in new_data.columns]
    for col in missing_cols:
        new_data[col] = None

    combined = pd.concat([current_df, new_data], ignore_index=True)
    combined = combined.drop_duplicates(subset="Sample", keep="first")
    combined = combined.sort_values("Sample").reset_index(drop=True)
    return combined

def extract_condition_components(df, condition_col="Sample"):
    df = df.copy()

    def parse_condition(name):
        if not isinstance(name, str):
            return pd.Series({
                "SampleID": None,
                "Age of Organoid": None,
                "Treatment": None,
                "Phase": None,
                "Time of Treatment": None
            })

        parts = name.split("_")
        return pd.Series({
            "SampleID": parts[0] if len(parts) > 0 else None,
            "Age of Organoid": parts[1] if len(parts) > 1 else None,
            "Treatment": parts[2] if len(parts) > 2 else None,
            "Phase": parts[3] if len(parts) > 3 else None,
            "Time of Treatment": parts[4] if len(parts) > 4 else None
        })

    components = df[condition_col].apply(parse_condition)
    df = pd.concat([df, components], axis=1)
    return df

def refresh_metrics_csv(datasets, csv_path):
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
    else:
        df = None

    trains = {k: sd.train for k, sd in datasets.items()}
    df = update_parkinsons_exp_metrics(trains, df)
    df = extract_condition_components(df, condition_col="Sample")  
    df.to_csv(csv_path, index=False)
    return df

def summarize_population_burst_metrics(datasets, config=None, time_start=0, time_window=180, plot=False):
    all_metrics = []
    per_dataset = {}

    for key, sd in datasets.items():
        print(f"Processing {key}...")
        analysis = BurstDetection(sd.train, config=config)
        result = analysis.compute_population_rate_and_bursts()

        if result[0] is None:
            print(f"[WARNING] Burst detection failed for '{key}'")
            continue

        times, smoothed, peaks, peak_times, bursts, burst_windows = result
        metrics = analysis.compute_pop_burst_metrics(
            times, smoothed, peaks, bursts,
            time_start=time_start,
            time_window=time_window,
            peak_times=peak_times,
            burst_windows=burst_windows
        )

        if plot:
            plot_raster(sd.train, title=f"{key} - Raster Plot")
            plot_population_rate(times, smoothed, bursts, title=f"{key} - Population Rate")

        if metrics:
            all_metrics.append(metrics)
            per_dataset[key] = metrics

    if not all_metrics:
        print("No datasets produced valid metrics.")
        return None

    def average_stat(stat_key):
        values = [m[stat_key] for m in all_metrics if stat_key in m]
        return {
            "mean": np.mean(values) if values else None,
            "std": np.std(values) if values else None
        }

    summary_keys = list(all_metrics[0].keys())
    summary = {k: average_stat(k) for k in summary_keys if isinstance(all_metrics[0][k], (int, float))}

    return {"overall": summary, "per_dataset": per_dataset}

def append_burst_metrics(datasets, metrics_func, csv_path, config=None):
    """
    Updates the Parkinsons experiment metrics CSV with new burst metrics.

    Parameters:
        datasets (dict): Dictionary of {dataset_key: SpikeDataset}
        metrics_func (callable): Function that computes metrics (e.g., compute_pop_burst_metrics)
        csv_path (str or Path): Path to the metrics CSV file
        config (dict): Optional config to pass to analysis function

    Returns:
        pd.DataFrame: Updated DataFrame with new metrics merged in
    """
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    new_entries = []

    for key, sd in datasets.items():
        detector = BurstDetection(sd.train, fs=sd.metadata.get("fs", 10000), config=config or {})
        times, smoothed, peaks, peak_times, bursts, burst_windows = detector.compute_population_rate_and_bursts()

        if times is None or not bursts:
            print(f"[WARNING] No bursts detected for '{key}'")
            continue

        metrics = detector.compute_pop_burst_metrics(
            times=times,
            smoothed=smoothed,
            peaks=peaks,
            bursts=bursts,
            time_start=config.get("time_start", 0),
            time_window=config.get("time_window", times[-1]),
            peak_times=peak_times,
            burst_windows=burst_windows
        )

        metrics["Sample"] = key
        new_entries.append(metrics)

    # Combine and merge
    if not new_entries:
        print("No metrics computed.")
        return df

    new_df = pd.DataFrame(new_entries)
    df = update_parkinsons_exp_metrics(
        trains_by_condition={row["Sample"]: None for _, row in new_df.iterrows()},
        current_df=df
    )

    # Merge in the new metrics
    df = pd.merge(df, new_df, on="Sample", how="outer")

    df.to_csv(csv_path, index=False)
    print(f"Updated burst metrics written to {csv_path}")
    return df