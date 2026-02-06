"""
Data Generation Script for Predictive Maintenance Project.

Generates:
  1. robots_combined_traindata.csv  - Clean training data (outliers removed)
  2. robots_combined_v2.csv         - Test data with subtle failure injection

Source: RMBR4-2_export_test.csv (Robot A raw current data, 8 axes)

Robot configurations:
  Robot A: 8 axes  |  Robot B: 10 axes  |  Robot C & D: 12 axes

Training: >= 10,000 points/robot  |  Testing: >= 50,000 points/robot
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Load and profile the raw data
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(SCRIPT_DIR, "RMBR4-2_export_test.csv")

print("=" * 70)
print("LOADING RAW DATA")
print("=" * 70)

df_raw = pd.read_csv(RAW_PATH)
print(f"Raw rows: {len(df_raw):,}")

# Rename axis columns to a simpler convention
axis_cols_raw = [c for c in df_raw.columns if c.startswith("Axis")]
axis_rename = {c: f"axis_{i+1}" for i, c in enumerate(axis_cols_raw)}
df_raw.rename(columns=axis_rename, inplace=True)
raw_axes = list(axis_rename.values())  # axis_1 .. axis_14
print(f"Axis columns found: {raw_axes}")

# Keep only the 8 axes that contain data (axes 9-14 are all NaN/empty)
ACTIVE_AXES = [f"axis_{i}" for i in range(1, 9)]
df_active = df_raw[ACTIVE_AXES].copy()

# ---------------------------------------------------------------------------
# 2. Identify "active" (non-idle) rows and compute clean statistics
# ---------------------------------------------------------------------------
# A row is idle if ALL 8 axes are 0
row_sums = df_active.sum(axis=1)
active_mask = row_sums > 0
df_active_only = df_active[active_mask].copy()
print(f"Active (non-idle) rows: {len(df_active_only):,}  "
      f"(idle: {(~active_mask).sum():,})")

# Per-axis stats on active rows (for outlier detection and normalisation)
axis_stats = {}
for ax in ACTIVE_AXES:
    vals = df_active_only[ax].dropna()
    axis_stats[ax] = {"mean": vals.mean(), "std": vals.std()}
    print(f"  {ax}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
          f"min={vals.min():.4f}  max={vals.max():.4f}")

# ---------------------------------------------------------------------------
# 3. Remove outliers (> 2.5 sigma from mean of active values)
# ---------------------------------------------------------------------------
OUTLIER_SIGMA = 2.5
print(f"\nRemoving outliers beyond {OUTLIER_SIGMA} sigma ...")

outlier_mask = pd.Series(False, index=df_active_only.index)
for ax in ACTIVE_AXES:
    s = axis_stats[ax]
    lo = s["mean"] - OUTLIER_SIGMA * s["std"]
    hi = s["mean"] + OUTLIER_SIGMA * s["std"]
    outlier_mask |= (df_active_only[ax] < lo) | (df_active_only[ax] > hi)

df_clean = df_active_only[~outlier_mask].copy()
print(f"Rows after outlier removal: {len(df_clean):,}  "
      f"(removed {outlier_mask.sum():,} outlier rows)")

# Recompute stats on clean data (these are used for z-score normalisation)
clean_stats = {}
for ax in ACTIVE_AXES:
    vals = df_clean[ax]
    clean_stats[ax] = {"mean": vals.mean(), "std": vals.std()}

# Z-score normalise the clean data
df_clean_norm = df_clean.copy()
for ax in ACTIVE_AXES:
    s = clean_stats[ax]
    df_clean_norm[ax] = (df_clean[ax] - s["mean"]) / s["std"]

clean_norm_values = df_clean_norm[ACTIVE_AXES].values  # numpy array
print(f"Clean normalised pool size: {len(clean_norm_values):,} rows x "
      f"{clean_norm_values.shape[1]} axes")

# Save normalisation params
params_df = pd.DataFrame([
    {"axis": ax, "mean": clean_stats[ax]["mean"], "std": clean_stats[ax]["std"]}
    for ax in ACTIVE_AXES
])
params_df.to_csv(os.path.join(SCRIPT_DIR, "normalization_params.csv"), index=False)
print("Saved normalization_params.csv")


# ---------------------------------------------------------------------------
# Helper: build a block of data by resampling from the clean pool
# ---------------------------------------------------------------------------
def build_resampled_block(n_points, n_axes, robot_offset_scale=0.05):
    """
    Resample *n_points* rows from the clean normalised pool.

    For axes 1-8  : resample directly from pool + small robot-specific offset.
    For axes 9-12 : synthesise from weighted combinations of axes 1-8.

    Returns an (n_points, 12) numpy array where unused axes are NaN.
    """
    pool = clean_norm_values  # (N, 8)
    n_pool = len(pool)

    # Resample with replacement to reach n_points
    indices = np.random.randint(0, n_pool, size=n_points)
    block = pool[indices].copy()  # (n_points, 8)

    # Add small per-robot variation so robots are not identical
    robot_shift = np.random.uniform(-robot_offset_scale, robot_offset_scale,
                                    size=(1, 8))
    block += robot_shift

    # Add tiny Gaussian jitter to break exact duplicates
    block += np.random.normal(0, 0.02, size=block.shape)

    # Build full 12-axis array
    full = np.full((n_points, 12), np.nan)
    full[:, :8] = block

    if n_axes >= 10:
        # Axis 9: weighted mix of axes 1 & 4 + noise
        full[:, 8] = 0.6 * block[:, 0] + 0.4 * block[:, 3] + \
                     np.random.normal(0, 0.15, n_points)
        # Axis 10: weighted mix of axes 2 & 5 + noise
        full[:, 9] = 0.5 * block[:, 1] + 0.5 * block[:, 4] + \
                     np.random.normal(0, 0.15, n_points)

    if n_axes >= 12:
        # Axis 11: weighted mix of axes 3 & 6 + noise
        full[:, 10] = 0.55 * block[:, 2] + 0.45 * block[:, 5] + \
                      np.random.normal(0, 0.15, n_points)
        # Axis 12: weighted mix of axes 7 & 8 + noise
        full[:, 11] = 0.65 * block[:, 6] + 0.35 * block[:, 7] + \
                      np.random.normal(0, 0.15, n_points)

    # NaN-out axes beyond the robot's configuration
    for i in range(n_axes, 12):
        full[:, i] = np.nan

    return full


# ---------------------------------------------------------------------------
# Helper: inject subtle failure degradation into a data block
# ---------------------------------------------------------------------------
def inject_failure(block, n_axes, failure_cfg):
    """
    Inject gradual degradation into the last portion of the data.

    failure_cfg dict keys:
      alert_start_frac  : fraction of total length where ALERT-level drift begins
      error_start_frac  : fraction where a brief ERROR-level spike occurs
      error_end_frac    : fraction where the ERROR spike ends
      alert_peak_sigma  : max drift magnitude in  sigma for ALERT region  (2.0 - 2.8)
      error_peak_sigma  : max drift magnitude in  sigma for ERROR spike   (3.0 - 3.5)
      affected_axes     : list of axis indices (0-based) to degrade
    """
    n = len(block)
    alert_start = int(n * failure_cfg["alert_start_frac"])
    error_start = int(n * failure_cfg["error_start_frac"])
    error_end   = int(n * failure_cfg["error_end_frac"])

    for ax_idx in failure_cfg["affected_axes"]:
        if ax_idx >= n_axes:
            continue

        # --- ALERT region: gradual ramp from 0 to alert_peak_sigma ---
        alert_len = error_start - alert_start
        if alert_len > 0:
            # Smooth ramp using half-cosine (starts slow, accelerates)
            t = np.linspace(0, np.pi / 2, alert_len)
            ramp = np.sin(t) * failure_cfg["alert_peak_sigma"]
            block[alert_start:error_start, ax_idx] += ramp

        # --- ERROR spike: brief region that exceeds 3 sigma ---
        error_len = error_end - error_start
        if error_len > 0:
            # Bell-shaped spike peaking at error_peak_sigma
            t = np.linspace(0, np.pi, error_len)
            spike = np.sin(t) * failure_cfg["error_peak_sigma"]
            block[error_start:error_end, ax_idx] += spike

        # --- Post-error: decay back to alert level then normal ---
        post_start = error_end
        post_len = n - post_start
        if post_len > 0:
            decay = np.linspace(failure_cfg["alert_peak_sigma"], 0, post_len)
            block[post_start:, ax_idx] += decay

    return block


# ===================================================================
#  GENERATE TRAINING DATA
# ===================================================================
print("\n" + "=" * 70)
print("GENERATING TRAINING DATA")
print("=" * 70)

ROBOT_CFG_TRAIN = {
    "Robot A": {"n_points": 10000, "n_axes": 8},
    "Robot B": {"n_points": 10000, "n_axes": 10},
    "Robot C": {"n_points": 10000, "n_axes": 12},
    "Robot D": {"n_points": 10000, "n_axes": 12},
}

ALL_AXES = [f"axis_{i}" for i in range(1, 13)]
train_frames = []
train_start = datetime(2024, 6, 1)

for robot_name, cfg in ROBOT_CFG_TRAIN.items():
    n, nax = cfg["n_points"], cfg["n_axes"]
    print(f"  {robot_name}: {n:,} points, {nax} axes")

    block = build_resampled_block(n, nax)

    # Build timestamps (1-second intervals)
    timestamps = [train_start + timedelta(seconds=i) for i in range(n)]

    df_robot = pd.DataFrame(block, columns=ALL_AXES)
    df_robot.insert(0, "robot", robot_name)
    df_robot.insert(0, "timestamp", timestamps)
    train_frames.append(df_robot)

df_train = pd.concat(train_frames, ignore_index=True)
train_path = os.path.join(SCRIPT_DIR, "robots_combined_traindata.csv")
df_train.to_csv(train_path, index=False)
print(f"\nSaved {train_path}")
print(f"  Total rows: {len(df_train):,}")
for rn in df_train["robot"].unique():
    subset = df_train[df_train["robot"] == rn]
    non_null = subset[ALL_AXES].notna().sum()
    active = non_null[non_null > 0].index.tolist()
    print(f"  {rn}: {len(subset):,} rows, active axes: {active}")


# ===================================================================
#  GENERATE TEST DATA
# ===================================================================
print("\n" + "=" * 70)
print("GENERATING TEST DATA")
print("=" * 70)

ROBOT_CFG_TEST = {
    "Robot A": {"n_points": 50000, "n_axes": 8},
    "Robot B": {"n_points": 50000, "n_axes": 10},
    "Robot C": {"n_points": 50000, "n_axes": 12},
    "Robot D": {"n_points": 50000, "n_axes": 12},
}

# Failure injection configurations per robot.
# Design: mostly ALERTs (2-2.8 sigma drift), only 1-2 brief ERROR spikes (>3 sigma).
# The drift starts late in the data (simulating "failure developing in ~2 weeks").
# Each robot degrades on different axes to show varied failure modes.
FAILURE_CFG = {
    "Robot A": {
        "alert_start_frac": 0.88,     # drift begins at 88% of data
        "error_start_frac": 0.96,     # brief error spike at 96%
        "error_end_frac":   0.965,    # error lasts 0.5% of data (~250 pts)
        "alert_peak_sigma": 2.5,      # ALERT-level drift peak
        "error_peak_sigma": 3.3,      # ERROR spike peak (just above 3 sigma)
        "affected_axes": [0, 2],      # axis_1, axis_3
    },
    "Robot B": {
        "alert_start_frac": 0.90,
        "error_start_frac": 0.97,
        "error_end_frac":   0.975,
        "alert_peak_sigma": 2.4,
        "error_peak_sigma": 3.2,
        "affected_axes": [1, 4, 8],   # axis_2, axis_5, axis_9
    },
    "Robot C": {
        "alert_start_frac": 0.87,
        "error_start_frac": 0.95,
        "error_end_frac":   0.955,
        "alert_peak_sigma": 2.6,
        "error_peak_sigma": 3.4,
        "affected_axes": [3, 9, 10],  # axis_4, axis_10, axis_11
    },
    "Robot D": {
        "alert_start_frac": 0.91,
        "error_start_frac": 0.975,
        "error_end_frac":   0.98,
        "alert_peak_sigma": 2.3,
        "error_peak_sigma": 3.1,
        "affected_axes": [5, 7, 11],  # axis_6, axis_8, axis_12
    },
}

test_frames = []
test_start = datetime(2024, 7, 1)

for robot_name, cfg in ROBOT_CFG_TEST.items():
    n, nax = cfg["n_points"], cfg["n_axes"]
    print(f"\n  {robot_name}: {n:,} points, {nax} axes")

    block = build_resampled_block(n, nax)

    # Inject failure degradation
    fcfg = FAILURE_CFG[robot_name]
    block = inject_failure(block, nax, fcfg)

    alert_start_idx = int(n * fcfg["alert_start_frac"])
    error_start_idx = int(n * fcfg["error_start_frac"])
    error_end_idx   = int(n * fcfg["error_end_frac"])
    print(f"    Failure injection:")
    print(f"      ALERT drift starts at index {alert_start_idx:,} "
          f"({fcfg['alert_start_frac']*100:.0f}%)")
    print(f"      ERROR spike: index {error_start_idx:,}-{error_end_idx:,} "
          f"({(error_end_idx-error_start_idx):,} points)")
    print(f"      Affected axes (0-indexed): {fcfg['affected_axes']}")
    print(f"      Alert peak: {fcfg['alert_peak_sigma']} sigma, "
          f"Error peak: {fcfg['error_peak_sigma']} sigma")

    timestamps = [test_start + timedelta(seconds=i) for i in range(n)]

    df_robot = pd.DataFrame(block, columns=ALL_AXES)
    df_robot.insert(0, "robot", robot_name)
    df_robot.insert(0, "timestamp", timestamps)
    test_frames.append(df_robot)

df_test = pd.concat(test_frames, ignore_index=True)
test_path = os.path.join(SCRIPT_DIR, "robots_combined_v2.csv")
df_test.to_csv(test_path, index=False)
print(f"\nSaved {test_path}")
print(f"  Total rows: {len(df_test):,}")
for rn in df_test["robot"].unique():
    subset = df_test[df_test["robot"] == rn]
    non_null = subset[ALL_AXES].notna().sum()
    active = non_null[non_null > 0].index.tolist()
    print(f"  {rn}: {len(subset):,} rows, active axes: {active}")


# ===================================================================
#  VALIDATION SUMMARY
# ===================================================================
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

print(f"\nTraining data: {train_path}")
print(f"  Shape: {df_train.shape}")
print(f"  Robots: {df_train['robot'].unique().tolist()}")
print(f"  Points per robot: "
      f"{df_train.groupby('robot').size().to_dict()}")

print(f"\nTest data: {test_path}")
print(f"  Shape: {df_test.shape}")
print(f"  Robots: {df_test['robot'].unique().tolist()}")
print(f"  Points per robot: "
      f"{df_test.groupby('robot').size().to_dict()}")

print(f"\nExpected alert system behaviour:")
print(f"  - Training data: clean, should produce NO alerts")
print(f"  - Test data: mostly clean with gradual drift near end")
print(f"    -> Expect ALERT events on degraded axes (sustained 2-3 sigma drift)")
print(f"    -> Expect only 1-2 ERROR events per robot (brief >3 sigma spikes)")
print(f"\nDone.")
