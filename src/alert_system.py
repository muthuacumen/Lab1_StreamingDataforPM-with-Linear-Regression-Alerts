"""
Alert System for Predictive Maintenance.

Implements:
- ALERT: deviation >= MinC sustained for >= T seconds
- ERROR: deviation >= MaxC sustained for >= T seconds
- Event logging to CSV and database
- Visualization with alert/error markers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import os
import csv


class AlertThresholds:
    """
    Configuration for alert thresholds.

    Attributes:
    -----------
    MinC : float
        Minimum deviation above regression line to trigger ALERT
    MaxC : float
        Maximum deviation above regression line to trigger ERROR
    T : int
        Minimum sustained duration in seconds
    """

    def __init__(self, MinC=2.0, MaxC=3.0, T=30):
        """
        Initialize thresholds.

        Parameters:
        -----------
        MinC : float
            Minimum deviation for ALERT (default: 2.0 * residual_std)
        MaxC : float
            Maximum deviation for ERROR (default: 3.0 * residual_std)
        T : int
            Minimum sustained duration in seconds (default: 30)
        """
        self.MinC = MinC
        self.MaxC = MaxC
        self.T = T

    def __repr__(self):
        return f"AlertThresholds(MinC={self.MinC}, MaxC={self.MaxC}, T={self.T}s)"


class AlertEvent:
    """
    Represents a single alert or error event.
    """

    def __init__(self, timestamp, axis, event_type, deviation,
                 duration_seconds, actual_value, predicted_value,
                 threshold_used, robot='Unknown'):
        self.timestamp = timestamp
        self.axis = axis
        self.event_type = event_type  # 'ALERT' or 'ERROR'
        self.deviation = deviation
        self.duration_seconds = duration_seconds
        self.actual_value = actual_value
        self.predicted_value = predicted_value
        self.threshold_used = threshold_used
        self.robot = robot

    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'axis': self.axis,
            'event_type': self.event_type,
            'deviation': self.deviation,
            'duration_seconds': self.duration_seconds,
            'actual_value': self.actual_value,
            'predicted_value': self.predicted_value,
            'threshold_used': self.threshold_used,
            'robot': self.robot
        }

    def __repr__(self):
        return (f"AlertEvent({self.event_type} on {self.axis} at {self.timestamp}, "
                f"deviation={self.deviation:.3f}, duration={self.duration_seconds}s)")


class AlertSystem:
    """
    Main alert detection and logging system.

    Monitors streaming data for sustained deviations from
    regression predictions and generates alerts/errors.
    """

    AXIS_NAMES = ['axis_1', 'axis_2', 'axis_3', 'axis_4',
                  'axis_5', 'axis_6', 'axis_7', 'axis_8',
                  'axis_9', 'axis_10', 'axis_11', 'axis_12']

    def __init__(self, regression_models, thresholds=None):
        """
        Initialize alert system.

        Parameters:
        -----------
        regression_models : RobotRegressionModels
            Trained regression models for all axes
        thresholds : AlertThresholds, optional
            Threshold configuration (uses defaults if None)
        """
        self.models = regression_models
        self.thresholds = thresholds or AlertThresholds()
        self.alert_log = []

        # Track sustained deviations for each axis
        self.deviation_trackers = {axis: {
            'start_time': None,
            'start_index': None,
            'level': 'NORMAL',  # 'NORMAL', 'ALERT', 'ERROR'
            'max_deviation': 0,
            'count': 0
        } for axis in self.AXIS_NAMES}

    def check_deviation(self, deviation, axis):
        """
        Check deviation level for a single value.

        Parameters:
        -----------
        deviation : float
            Deviation value (actual - predicted)
        axis : str
            Axis name

        Returns:
        --------
        str : 'NORMAL', 'ALERT', or 'ERROR'
        """
        # Get axis-specific residual std for threshold scaling
        if self.models.models[axis].is_fitted:
            residual_std = self.models.models[axis].residual_std
        else:
            residual_std = 1.0

        # Scale thresholds by residual std
        alert_threshold = self.thresholds.MinC * residual_std
        error_threshold = self.thresholds.MaxC * residual_std

        if abs(deviation) >= error_threshold:
            return 'ERROR'
        elif abs(deviation) >= alert_threshold:
            return 'ALERT'
        else:
            return 'NORMAL'

    def process_streaming_data(self, df_test, time_interval_seconds=1):
        """
        Process entire test dataset for alerts.

        Parameters:
        -----------
        df_test : DataFrame
            Test data to process
        time_interval_seconds : int
            Time interval between data points

        Returns:
        --------
        list : List of AlertEvent objects
        """
        print("=" * 60)
        print("PROCESSING STREAMING DATA FOR ALERTS")
        print("=" * 60)
        print(f"Thresholds: {self.thresholds}")
        print(f"Processing {len(df_test):,} records...")

        time_index = np.arange(len(df_test))
        alerts_triggered = []

        # Get predictions for all axes
        predictions = self.models.predict_all_axes(df_test)

        for idx in range(len(df_test)):
            current_time = df_test.iloc[idx].get('timestamp', idx)
            robot = df_test.iloc[idx].get('robot', 'Unknown')

            for axis in self.AXIS_NAMES:
                if axis not in df_test.columns:
                    continue
                if not self.models.models[axis].is_fitted:
                    continue

                actual = df_test.iloc[idx][axis]
                predicted = predictions.iloc[idx][f'{axis}_predicted']

                if pd.isna(actual):
                    continue

                deviation = actual - predicted
                level = self.check_deviation(deviation, axis)

                tracker = self.deviation_trackers[axis]

                if level in ['ALERT', 'ERROR']:
                    # Start or continue tracking
                    if tracker['level'] == 'NORMAL':
                        tracker['start_time'] = current_time
                        tracker['start_index'] = idx
                        tracker['max_deviation'] = abs(deviation)
                        tracker['count'] = 1
                    else:
                        tracker['count'] += 1
                        tracker['max_deviation'] = max(tracker['max_deviation'], abs(deviation))

                    tracker['level'] = level

                    # Check if sustained for T seconds
                    duration = tracker['count'] * time_interval_seconds
                    if duration >= self.thresholds.T:
                        # Check if we haven't already logged this event
                        if not self._event_already_logged(axis, tracker['start_index']):
                            event = AlertEvent(
                                timestamp=tracker['start_time'],
                                axis=axis,
                                event_type=level,
                                deviation=tracker['max_deviation'],
                                duration_seconds=duration,
                                actual_value=actual,
                                predicted_value=predicted,
                                threshold_used=self.thresholds.MinC if level == 'ALERT' else self.thresholds.MaxC,
                                robot=robot
                            )
                            alerts_triggered.append(event)
                            self.alert_log.append(event)
                else:
                    # Reset tracker
                    tracker['level'] = 'NORMAL'
                    tracker['start_time'] = None
                    tracker['start_index'] = None
                    tracker['max_deviation'] = 0
                    tracker['count'] = 0

            # Progress update
            if (idx + 1) % 10000 == 0:
                print(f"  Processed {idx + 1:,} / {len(df_test):,} records, "
                      f"{len(self.alert_log)} events logged")

        print(f"\nProcessing complete: {len(self.alert_log)} total events logged")
        return alerts_triggered

    def _event_already_logged(self, axis, start_index):
        """Check if event at this position already logged."""
        for event in self.alert_log:
            if event.axis == axis:
                # Simple check - could be more sophisticated
                if hasattr(event, '_start_index') and event._start_index == start_index:
                    return True
        return False

    def save_log_to_csv(self, path='logs/alert_log.csv'):
        """
        Save alert log to CSV file.

        Parameters:
        -----------
        path : str
            Path to save CSV file
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w', newline='') as f:
            fieldnames = ['timestamp', 'axis', 'event_type', 'deviation',
                         'duration_seconds', 'actual_value', 'predicted_value',
                         'threshold_used', 'robot']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for event in self.alert_log:
                writer.writerow(event.to_dict())

        print(f"Alert log saved to: {path} ({len(self.alert_log)} events)")

    def get_alert_summary(self):
        """
        Get summary statistics of logged alerts.

        Returns:
        --------
        DataFrame : Summary statistics
        """
        if not self.alert_log:
            return pd.DataFrame()

        df = pd.DataFrame([e.to_dict() for e in self.alert_log])

        summary = df.groupby(['axis', 'event_type']).agg({
            'deviation': ['count', 'mean', 'max'],
            'duration_seconds': ['mean', 'max']
        }).round(3)

        return summary


    def generate_alert_dashboard(self, df_test, save_path='alerts/dashboard.png'):
        """
        Generate comprehensive dashboard with all 12 axes.

        Parameters:
        -----------
        df_test : DataFrame
            Test data
        save_path : str
            Path to save dashboard
        """
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        fig.suptitle('Predictive Maintenance Alert Dashboard - All 12 Axes',
                     fontsize=18, fontweight='bold')

        time_index = np.arange(len(df_test))

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
                  '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2',
                  '#F06292', '#AED581', '#FFD54F', '#90CAF9']

        for idx, axis_name in enumerate(self.AXIS_NAMES):
            row = idx // 4
            col = idx % 4
            ax = axes[row, col]

            # Check if axis has data
            if axis_name not in df_test.columns:
                ax.set_title(f'{axis_name} (Not Found)', fontsize=11, fontweight='bold')
                ax.text(0.5, 0.5, 'Column Not Found', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_xlabel('Time', fontsize=9)
                ax.set_ylabel('Current', fontsize=9)
                ax.grid(True, alpha=0.3)
                continue

            valid_data = df_test[axis_name].dropna()
            if len(valid_data) == 0:
                ax.set_title(f'{axis_name} (No Data)', fontsize=11, fontweight='bold')
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_xlabel('Time', fontsize=9)
                ax.set_ylabel('Current', fontsize=9)
                ax.grid(True, alpha=0.3)
                continue

            # Plot actual data
            ax.plot(time_index, df_test[axis_name], alpha=0.5, linewidth=0.3,
                   color=colors[idx])

            # Plot regression line
            if self.models.models[axis_name].is_fitted:
                predictions = self.models.models[axis_name].predict(time_index)
                ax.plot(time_index, predictions, color='black', linewidth=1.5,
                       linestyle='--', alpha=0.7)

            # Count events for this axis
            axis_events = [e for e in self.alert_log if e.axis == axis_name]
            alert_count = len([e for e in axis_events if e.event_type == 'ALERT'])
            error_count = len([e for e in axis_events if e.event_type == 'ERROR'])

            # Mark alerts
            for event in axis_events[:20]:  # Limit markers to avoid clutter
                color = 'orange' if event.event_type == 'ALERT' else 'red'
                marker = '^' if event.event_type == 'ALERT' else 'X'
                ax.scatter([0], [event.actual_value], color=color, marker=marker,
                          s=50, zorder=5, alpha=0.7)

            ax.set_title(f'{axis_name} (A:{alert_count}, E:{error_count})',
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Time', fontsize=9)
            ax.set_ylabel('Current', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

        # Add legend
        alert_patch = mpatches.Patch(color='orange', label='ALERT')
        error_patch = mpatches.Patch(color='red', label='ERROR')
        fig.legend(handles=[alert_patch, error_patch], loc='lower center',
                  ncol=2, fontsize=12)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Dashboard saved to: {save_path}")

        return fig


# Test the module
if __name__ == '__main__':
    print("Testing Alert System...")

    # Create mock regression models
    from linear_regression_model import RobotRegressionModels
    import numpy as np

    np.random.seed(42)
    n_samples = 1000

    df_test = pd.DataFrame({
        'timestamp': pd.date_range('2024-06-01', periods=n_samples, freq='S'),
        'robot': 'Robot A',
        'axis_1': np.concatenate([np.random.randn(800), np.random.randn(200) + 3]),
        'axis_2': np.random.randn(n_samples),
    })

    models = RobotRegressionModels()
    models.train_all_axes(df_test)

    # Test alert system
    thresholds = AlertThresholds(MinC=2.0, MaxC=3.0, T=5)
    alert_system = AlertSystem(models, thresholds)

    alerts = alert_system.process_streaming_data(df_test)

    print(f"\nTotal alerts: {len(alerts)}")
    print("\nAlert Summary:")
    print(alert_system.get_alert_summary())

    print("\nAlert System test complete")
