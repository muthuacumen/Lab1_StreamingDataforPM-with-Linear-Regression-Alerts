"""
Analyze deviations from linear regression predictions for each robot+axis.
Compute consecutive run lengths at ALERT (4*std) and ERROR (5*std) levels.
Determine minimum T values and event counts for various T.
"""

import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
data_path = r"C:\Projects\Lab1_StreamingDataforPMwithLinRegAlerts\data\robots_combined_v2.csv"
params_path = r"C:\Projects\Lab1_StreamingDataforPMwithLinRegAlerts\data\model_params.csv"

df = pd.read_csv(data_path)
params = pd.read_csv(params_path)

# Only fitted axes
fitted_params = params[params["is_fitted"] == True].copy()

print(f"Test data shape: {df.shape}")
print(f"Robots in data: {df['robot'].unique()}")
print(f"Fitted model params: {len(fitted_params)} rows")
print()

# ------------------------------------------------------------------
# Helper: find consecutive runs where condition is True
# ------------------------------------------------------------------
def consecutive_run_lengths(bool_array):
    runs = []
    count = 0
    for v in bool_array:
        if v:
            count += 1
        else:
            if count > 0:
                runs.append(count)
            count = 0
    if count > 0:
        runs.append(count)
    return runs

# ------------------------------------------------------------------
# 2-4. Per robot+axis analysis
# ------------------------------------------------------------------
results = []
robots = sorted(df["robot"].unique())

for robot in robots:
    robot_df = df[df["robot"] == robot].copy()
    robot_df = robot_df.sort_values("timestamp").reset_index(drop=True)
    robot_df = robot_df.dropna(axis=1, how="all")
    remaining_axis_cols = [c for c in robot_df.columns if c.startswith("axis_")]
    N = len(robot_df)
    time_index = np.arange(N, dtype=float)
    robot_params = fitted_params[fitted_params["robot"] == robot]

    for _, row in robot_params.iterrows():
        axis = row["axis"]
        if axis not in remaining_axis_cols:
            continue
        slope = row["slope"]
        intercept = row["intercept"]
        residual_std = row["residual_std"]
        actual = robot_df[axis].values.astype(float)
        predicted = slope * time_index + intercept
        deviation = actual - predicted
        alert_thresh = 4.0 * residual_std
        error_thresh = 5.0 * residual_std
        abs_dev = np.abs(deviation)
        alert_mask = abs_dev >= alert_thresh
        error_mask = abs_dev >= error_thresh
        alert_runs = consecutive_run_lengths(alert_mask)
        error_runs = consecutive_run_lengths(error_mask)
        max_alert_run = max(alert_runs) if alert_runs else 0
        max_error_run = max(error_runs) if error_runs else 0
        n_alert_points = int(alert_mask.sum())
        n_error_points = int(error_mask.sum())
        results.append({
            "robot": robot, "axis": axis,
            "residual_std": residual_std,
            "alert_threshold": alert_thresh,
            "error_threshold": error_thresh,
            "max_consec_alert": max_alert_run,
            "max_consec_error": max_error_run,
            "n_alert_points": n_alert_points,
            "n_error_points": n_error_points,
            "alert_runs": alert_runs,
            "error_runs": error_runs,
        })

# Print table
print("=" * 130)
print(f"{'Robot':<10} {'Axis':<8} {'Resid Std':>10} {'Alert Thr':>10} {'Error Thr':>10} "
      f"{'MaxRun AL':>10} {'MaxRun ER':>10} {'#Pts AL':>10} {'#Pts ER':>10}")
print("=" * 130)
for r in results:
    print(f"{r['robot']:<10} {r['axis']:<8} {r['residual_std']:>10.6f} {r['alert_threshold']:>10.6f} "
          f"{r['error_threshold']:>10.6f} {r['max_consec_alert']:>10d} {r['max_consec_error']:>10d} "
          f"{r['n_alert_points']:>10d} {r['n_error_points']:>10d}")
print("=" * 130)
print()

# Step 5
print("=" * 90)
print("STEP 5: Consecutive Run Analysis")
print("=" * 90)
global_max_alert = max(r["max_consec_alert"] for r in results)
global_max_error = max(r["max_consec_error"] for r in results)
print(f"\nGlobal max consecutive run at ALERT level: {global_max_alert}")
print(f"Global max consecutive run at ERROR level: {global_max_error}")
print()
print(f"Minimum T that produces at least one ALERT event: T=1 (any T from 1 to {global_max_alert} works)")
print(f"Minimum T that produces at least one ERROR event: T=1 (any T from 1 to {global_max_error} works)")
print()
print(f"Maximum T that still produces at least one ALERT event: T={global_max_alert}")
print(f"Maximum T that still produces at least one ERROR event: T={global_max_error}")
print()

T_values = [5, 10, 15, 20, 25, 30]
print("-" * 90)
print(f"{'T':>5}  {'Total ALERT events':>20}  {'Total ERROR events':>20}")
print("-" * 90)
for T in T_values:
    total_alert = sum(sum(1 for run in r["alert_runs"] if run >= T) for r in results)
    total_error = sum(sum(1 for run in r["error_runs"] if run >= T) for r in results)
    print(f"{T:>5}  {total_alert:>20}  {total_error:>20}")
print("-" * 90)
print()

print("=" * 90)
print("DETAILED: Events per robot+axis for each T value")
print("=" * 90)
for T in T_values:
    print(f"\n--- T = {T} ---")
    print(f"  {'Robot':<10} {'Axis':<8} {'ALERT events':>14} {'ERROR events':>14}")
    any_events = False
    for r in results:
        a_ev = sum(1 for run in r["alert_runs"] if run >= T)
        e_ev = sum(1 for run in r["error_runs"] if run >= T)
        if a_ev > 0 or e_ev > 0:
            print(f"  {r['robot']:<10} {r['axis']:<8} {a_ev:>14} {e_ev:>14}")
            any_events = True
    if not any_events:
        print("  (no events at this T)")

print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"With standard normal residuals (~std=1.0):")
print(f"  ALERT threshold (4*std) ~= 4.0  ->  P(|Z|>=4) ~= 0.0063%")
print(f"  ERROR threshold (5*std) ~= 5.0  ->  P(|Z|>=5) ~= 0.000057%")
print(f"  Expected alert points per 50000: ~3.2")
print(f"  Expected error points per 50000: ~0.03")
print()
total_alert_pts = sum(r["n_alert_points"] for r in results)
total_error_pts = sum(r["n_error_points"] for r in results)
total_axes = len(results)
print(f"Actual observations across all robots+axes:")
print(f"  Total fitted axes: {total_axes}")
print(f"  Total alert-level points: {total_alert_pts} (avg {total_alert_pts/total_axes:.1f} per axis)")
print(f"  Total error-level points: {total_error_pts} (avg {total_error_pts/total_axes:.1f} per axis)")
