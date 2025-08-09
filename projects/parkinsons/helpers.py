
# --- Persistence helpers for metrics_df, histogram_data, and burst_features_df ---

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ---------- JSON parsing helpers ----------
def _safe_json_load(x):
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return x
    return x

def _jsonify_list(v):
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return json.dumps(list(v))
    return v

# ---------- METRICS: save/load ----------
def save_metrics_df(metrics_df, base_dir="~/bioinformatics/projects/parkinsons"):
    base = Path(base_dir).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    csv_path = base / "metrics.csv"

    if metrics_df is None or metrics_df.empty:
        print("metrics_df is empty — nothing saved.")
        return csv_path

    df = metrics_df.copy()
    # JSON-serialize list-like columns if present
    for col in ["peak_times", "burst_windows"]:
        if col in df.columns:
            df[col] = df[col].apply(_jsonify_list)

    df.to_csv(csv_path, index=False)
    print(f"Saved metrics_df to {csv_path} ({len(df)} rows)")
    return csv_path

def load_metrics_df(base_dir="~/bioinformatics/projects/parkinsons"):
    csv_path = Path(base_dir).expanduser() / "metrics.csv"
    if not csv_path.exists():
        print(f"metrics.csv not found at {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(_safe_json_load)
    print(f"Loaded metrics_df from {csv_path} ({len(df)} rows)")
    return df

# ---------- HISTOGRAM: save/append/load ----------
def save_histogram_data(histogram_data, base_dir="~/bioinformatics/projects/parkinsons"):
    """
    histogram_data: list of dicts like:
        {"Dataset": str, "Group": str, "RelPeaks": np.ndarray or list}
    Saves both CSV and Parquet.
    """
    base = Path(base_dir).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    csv_path = base / "histogram_data.csv"
    pq_path = base / "histogram_data.parquet"

    if not histogram_data:
        print("histogram_data empty — nothing saved.")
        return csv_path, pq_path

    rows = []
    for entry in histogram_data:
        rel = entry.get("RelPeaks", [])
        if isinstance(rel, np.ndarray):
            rel = rel.tolist()
        rows.append({
            "Dataset": entry.get("Dataset"),
            "Group": entry.get("Group"),
            "RelPeaks": json.dumps(rel),
            "N": len(rel),
            "Min": (min(rel) if len(rel) else None),
            "Max": (max(rel) if len(rel) else None),
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(pq_path, index=False)
    except Exception as e:
        print(f"ℹ️ Parquet save skipped: {e}")

    print(f"Saved histogram_data to {csv_path} ({len(df)} datasets)")
    return csv_path, pq_path

def append_histogram_entry(entry, base_dir="~/bioinformatics/projects/parkinsons"):
    """
    Upsert a single histogram entry (replace same Dataset if exists)
    """
    base = Path(base_dir).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    csv_path = base / "histogram_data.csv"
    pq_path = base / "histogram_data.parquet"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["Dataset", "Group", "RelPeaks", "N", "Min", "Max"])

    rel = entry.get("RelPeaks", [])
    if isinstance(rel, np.ndarray):
        rel = rel.tolist()

    new_row = {
        "Dataset": entry.get("Dataset"),
        "Group": entry.get("Group"),
        "RelPeaks": json.dumps(rel),
        "N": len(rel),
        "Min": (min(rel) if len(rel) else None),
        "Max": (max(rel) if len(rel) else None),
    }

    df = df[df["Dataset"] != new_row["Dataset"]]
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(pq_path, index=False)
    except Exception as e:
        print(f"Parquet save skipped: {e}")

    print(f"Saved histogram_data entry for {new_row['Dataset']} ({len(df)} total datasets)")
    return csv_path, pq_path

def load_histogram_data(base_dir="~/bioinformatics/projects/parkinsons"):
    """
    Returns both:
      - df (CSV/parquet as DataFrame)
      - histogram_data list-of-dicts with RelPeaks as numpy arrays for analysis
    """
    base = Path(base_dir).expanduser()
    csv_path = base / "histogram_data.csv"
    pq_path = base / "histogram_data.parquet"

    if pq_path.exists():
        df = pd.read_parquet(pq_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        print("No histogram_data file found")
        return pd.DataFrame(), []

    # Parse RelPeaks JSON
    parsed = []
    for _, row in df.iterrows():
        rel = row["RelPeaks"]
        rel = json.loads(rel) if isinstance(rel, str) else rel
        parsed.append({
            "Dataset": row["Dataset"],
            "Group": row.get("Group"),
            "RelPeaks": np.array(rel, dtype=float) if rel is not None else np.array([]),
        })

    print(f"Loaded histogram_data ({len(df)} datasets)")
    return df, parsed

# ---------- BURST FEATURES: save/load ----------
def save_burst_features_df(burst_features_df, base_dir="~/bioinformatics/projects/parkinsons"):
    base = Path(base_dir).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    csv_path = base / "burst_features_full_summary.csv"
    if burst_features_df is None or burst_features_df.empty:
        print("burst_features_df is empty — nothing saved.")
        return csv_path
    burst_features_df.to_csv(csv_path, index=False)
    print(f"Saved burst_features_df to {csv_path} ({len(burst_features_df)} rows)")
    return csv_path

def load_burst_features_df(base_dir="~/bioinformatics/projects/parkinsons"):
    csv_path = Path(base_dir).expanduser() / "burst_features_full_summary.csv"
    if not csv_path.exists():
        print(f"burst_features_full_summary.csv not found at {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    print(f"Loaded burst_features_df from {csv_path} ({len(df)} rows)")
    return df

# ---------- One-shot: persist everything in memory ----------
def persist_session(metrics_df=None, histogram_data=None, burst_features_df=None,
                    base_dir="~/bioinformatics/projects/parkinsons"):
    if metrics_df is not None:
        save_metrics_df(metrics_df, base_dir=base_dir)
    if histogram_data is not None:
        save_histogram_data(histogram_data, base_dir=base_dir)
    if burst_features_df is not None:
        save_burst_features_df(burst_features_df, base_dir=base_dir)
    print("Session artifacts saved.")

# ---------- One-shot: reload everything ----------
def reload_session(base_dir="~/bioinformatics/projects/parkinsons"):
    m = load_metrics_df(base_dir=base_dir)
    h_df, h_list = load_histogram_data(base_dir=base_dir)
    b = load_burst_features_df(base_dir=base_dir)
    return {
        "metrics_df": m,
        "histogram_data_df": h_df,
        "histogram_data_list": h_list,
        "burst_features_df": b
    }

# ------------------------------------
#        Data Integrity Checks
# ------------------------------------

# ---------- Quick audit ----------
def audit_session_vars():
    print("DataFrame audit:")
    print("metrics_df:", len(metrics_df) if 'metrics_df' in globals() and isinstance(metrics_df, pd.DataFrame) else "not found")
    print("histogram_data:", len(histogram_data) if 'histogram_data' in globals() and isinstance(histogram_data, list) else "not found")
    print("burst_features_df:", burst_features_df.shape if 'burst_features_df' in globals() and isinstance(burst_features_df, pd.DataFrame) else "not found")


def integrity_check_dataframes(main_df=None, burst_df=None, histogram_df=None):
    """
    Detects duplicates (Dataset IDs repeated)
    Warns if some SampleIDs have fewer timepoints than others (possible missing recordings)
    Cross-checks consistency between histogram and burst features dataframes
    Flags redundant rows that could skew plots or stats
    Produces a clean report (passed / issues found)

    Performs integrity checks across multiple dataframes to ensure:
    - No duplicate Dataset IDs
    - No missing timepoints within each SampleID
    - Consistency of SampleIDs across different dataframes
    - No redundant rows that could bias plots or stats
    """

    issues = []

    # --- Check duplicates within each dataframe ---
    for name, df in {
        "Main DF": main_df,
        "Burst Features DF": burst_df,
        "Histogram DF": histogram_df
    }.items():
        if df is not None:
            dupes = df[df.duplicated(subset=["Dataset"])]
            if not dupes.empty:
                issues.append(f"{name} contains {len(dupes)} duplicate Dataset entries.")
            else:
                print(f"No duplicate Dataset IDs found in {name}")

    # --- Check for missing timepoints per SampleID ---
    if main_df is not None and "SampleID" in main_df.columns and "Timepoint" in main_df.columns:
        timepoint_counts = main_df.groupby("SampleID")["Timepoint"].nunique()
        missing_tp = timepoint_counts[timepoint_counts < timepoint_counts.max()]
        if not missing_tp.empty:
            issues.append(f"Some SampleIDs have fewer timepoints: {missing_tp.to_dict()}")
        else:
            print("All SampleIDs have consistent timepoints in Main DF")

    # --- Cross-reference SampleIDs between dataframes ---
    if burst_df is not None and histogram_df is not None:
        burst_ids = set(burst_df["SampleID"].unique())
        hist_ids = set(histogram_df["SampleID"].unique())
        missing_in_burst = hist_ids - burst_ids
        missing_in_hist = burst_ids - hist_ids
        if missing_in_burst:
            issues.append(f" {len(missing_in_burst)} SampleIDs are in Histogram DF but missing in Burst Features DF: {missing_in_burst}")
        if missing_in_hist:
            issues.append(f" {len(missing_in_hist)} SampleIDs are in Burst Features DF but missing in Histogram DF: {missing_in_hist}")
        if not missing_in_burst and not missing_in_hist:
            print("SampleIDs match between Burst Features and Histogram DF")

    # --- Check redundant rows (Dataset + Timepoint duplicates) ---
    for name, df in {
        "Main DF": main_df,
        "Burst Features DF": burst_df,
        "Histogram DF": histogram_df
    }.items():
        if df is not None and {"Dataset", "Timepoint"} <= set(df.columns):
            redund = df[df.duplicated(subset=["Dataset", "Timepoint"])]
            if not redund.empty:
                issues.append(f"{name} has redundant rows for the same Dataset/Timepoint combination.")

    print("\n=== Integrity Check Report ===")
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("All checks passed. Dataframes appear consistent and ready for analysis.")


# ---------------- JSON helpers ----------------
def _safe_json_load(x):
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return x
    return x

def _jsonify_list(v):
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return json.dumps(list(v))
    return v

# ---------------- METRICS save/load ----------------

def save_metrics_df(metrics_df, base_dir="~/bioinformatics/projects/parkinsons"):
    base = Path(base_dir).expanduser(); base.mkdir(parents=True, exist_ok=True)
    p = base / "metrics.csv"
    if metrics_df is None or metrics_df.empty:
        print("metrics_df is empty — nothing saved."); return p
    df = metrics_df.copy()
    for col in ["peak_times", "burst_windows"]:
        if col in df.columns:
            df[col] = df[col].apply(_jsonify_list)
    df.to_csv(p, index=False)
    print(f"Saved metrics_df to {p} ({len(df)} rows)")
    return p

def load_metrics_df(base_dir="~/bioinformatics/projects/parkinsons"):
    p = Path(base_dir).expanduser() / "metrics.csv"
    if not p.exists():
        print(f"metrics.csv not found at {p}"); return pd.DataFrame()
    df = pd.read_csv(p)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(_safe_json_load)
    print(f"Loaded metrics_df from {p} ({len(df)} rows)")
    return df

# ---------------- HISTOGRAM save/load ----------------

def save_histogram_data(histogram_data, base_dir="~/bioinformatics/projects/parkinsons"):
    base = Path(base_dir).expanduser(); base.mkdir(parents=True, exist_ok=True)
    csv_path = base / "histogram_data.csv"; pq_path = base / "histogram_data.parquet"
    if not histogram_data:
        print("histogram_data empty — nothing saved."); return csv_path, pq_path
    rows = []
    for entry in histogram_data:
        rel = entry.get("RelPeaks", [])
        if isinstance(rel, np.ndarray): rel = rel.tolist()
        rows.append({
            "Dataset": entry.get("Dataset"),
            "Group": entry.get("Group"),
            "RelPeaks": json.dumps(rel),
            "N": len(rel),
            "Min": (min(rel) if len(rel) else None),
            "Max": (max(rel) if len(rel) else None),
        })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    try: df.to_parquet(pq_path, index=False)
    except Exception as e: print(f"ℹ️ Parquet save skipped: {e}")
    print(f"Saved histogram_data to {csv_path} ({len(df)} datasets)")
    return csv_path, pq_path

def load_histogram_data(base_dir="~/bioinformatics/projects/parkinsons"):
    base = Path(base_dir).expanduser()
    csv_path = base / "histogram_data.csv"; pq_path = base / "histogram_data.parquet"
    if pq_path.exists():
        df = pd.read_parquet(pq_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        print("No histogram_data file found"); return pd.DataFrame(), []
    parsed = []
    for _, row in df.iterrows():
        rel = row["RelPeaks"]; rel = json.loads(rel) if isinstance(rel, str) else rel
        parsed.append({"Dataset": row["Dataset"], "Group": row.get("Group"),
                       "RelPeaks": np.array(rel, dtype=float) if rel is not None else np.array([])})
    print(f"Loaded histogram_data ({len(df)} datasets)")
    return df, parsed

# ---------------- BURST FEATURES save/load ----------------

def save_burst_features_df(burst_features_df, base_dir="~/bioinformatics/projects/parkinsons"):
    base = Path(base_dir).expanduser(); base.mkdir(parents=True, exist_ok=True)
    p = base / "burst_features_full_summary.csv"
    if burst_features_df is None or burst_features_df.empty:
        print("burst_features_df is empty — nothing saved."); return p
    burst_features_df.to_csv(p, index=False)
    print(f"Saved burst_features_df to {p} ({len(burst_features_df)} rows)")
    return p

def load_burst_features_df(base_dir="~/bioinformatics/projects/parkinsons"):
    p = Path(base_dir).expanduser() / "burst_features_full_summary.csv"
    if not p.exists():
        print(f"burst_features_full_summary.csv not found at {p}"); return pd.DataFrame()
    df = pd.read_csv(p)
    print(f"Loaded burst_features_df from {p} ({len(df)} rows)")
    return df

# ---------------- Session persist/reload ----------------

def persist_session(metrics_df=None, histogram_data=None, burst_features_df=None,
                    base_dir="~/bioinformatics/projects/parkinsons"):
    if metrics_df is not None: save_metrics_df(metrics_df, base_dir=base_dir)
    if histogram_data is not None: save_histogram_data(histogram_data, base_dir=base_dir)
    if burst_features_df is not None: save_burst_features_df(burst_features_df, base_dir=base_dir)
    print("Session artifacts saved.")

def reload_session(base_dir="~/bioinformatics/projects/parkinsons"):
    m = load_metrics_df(base_dir=base_dir)
    h_df, h_list = load_histogram_data(base_dir=base_dir)
    b = load_burst_features_df(base_dir=base_dir)
    return {"metrics_df": m, "histogram_data_df": h_df, "histogram_data_list": h_list, "burst_features_df": b}

def audit_session_vars(globals_):
    """Pass globals() here to see what’s loaded."""
    print("DataFrame audit:")
    md = globals_.get("metrics_df", None)
    hd = globals_.get("histogram_data", None)
    bf = globals_.get("burst_features_df", None)
    print("  metrics_df:", len(md) if isinstance(md, pd.DataFrame) else "not found")
    print("  histogram_data:", len(hd) if isinstance(hd, list) else "not found")
    print("  burst_features_df:", bf.shape if isinstance(bf, pd.DataFrame) else "not found")

# ---------------- Batch burst-features helper ----------------

def compute_and_save_burst_features(orc, dataset_groups, config, histogram_data, save_path="burst_features_combined.csv"):
    """Compute burst features for each dataset, add Group, save combined CSV; returns (dataset_dict, df)."""
    dataset_dict = {}; all_results = []
    print("Computing burst features for all dataset groups...")
    for group_name, datasets in dataset_groups.items():
        for dataset_key in datasets:
            try:
                df = compute_burst_features_full(orc, {dataset_key: [dataset_key]}, config, histogram_data)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df["Group"] = group_name
                    all_results.append(df); dataset_dict[dataset_key] = df
                    print(f"Processed: {dataset_key} ({len(df)} rows)")
                else:
                    print(f"No data returned for {dataset_key}")
            except Exception as e:
                print(f"Error processing {dataset_key}: {e}")
    if all_results:
        out = pd.concat(all_results, ignore_index=True)
        out["SampleID"] = out["Dataset"].str.split('_').str[0]
        out.to_csv(save_path, index=False)
        print(f"\nSaved all burst features to {save_path}")
        print(f"Final dataframe shape: {out.shape}")
        print("Sample IDs:", out["SampleID"].unique())
    else:
        out = pd.DataFrame(); print("No burst features computed.")
    return dataset_dict, out
