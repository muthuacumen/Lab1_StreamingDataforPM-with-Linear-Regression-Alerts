"""
Linear Regression Models for Predictive Maintenance.

Provides:
- Per-axis univariate linear regression (Time → Axis value)
- Model parameter storage (slope, intercept)
- Residual analysis
- Visualization utilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os


class AxisRegressionModel:
    """
    Linear regression model for a single robot axis.

    Fits: y = slope * time_index + intercept
    Where time_index is a sequential integer representing time.
    """

    def __init__(self, axis_name):
        """
        Initialize model for specified axis.

        Parameters:
        -----------
        axis_name : str
            Name of the axis (e.g., 'axis_1')
        """
        self.axis_name = axis_name
        self.slope = None
        self.intercept = None
        self.residual_mean = None
        self.residual_std = None
        self.model = LinearRegression()
        self.is_fitted = False

    def fit(self, time_index, axis_values):
        """
        Fit linear regression model.

        Parameters:
        -----------
        time_index : array-like
            Sequential time indices (e.g., 0, 1, 2, ...)
        axis_values : array-like
            Axis current values
        """
        # Reshape for sklearn
        X = np.array(time_index).reshape(-1, 1)
        y = pd.to_numeric(pd.Series(axis_values), errors='coerce').values

        # Remove NaN values
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        if len(y) < 10:
            print(f"Warning: {self.axis_name} has insufficient data ({len(y)} points)")
            return

        # Fit model
        self.model.fit(X, y)

        # Store parameters
        self.slope = self.model.coef_[0]
        self.intercept = self.model.intercept_

        # Compute residual statistics
        predictions = self.model.predict(X)
        residuals = y - predictions
        self.residual_mean = np.mean(residuals)
        self.residual_std = np.std(residuals)

        self.is_fitted = True

    def predict(self, time_index):
        """
        Make predictions for given time indices.

        Parameters:
        -----------
        time_index : array-like
            Time indices for prediction

        Returns:
        --------
        array : Predicted values
        """
        if not self.is_fitted:
            raise ValueError(f"Model for {self.axis_name} not fitted yet")

        X = np.array(time_index).reshape(-1, 1)
        return self.model.predict(X)

    def get_residuals(self, time_index, axis_values):
        """
        Compute residuals (observed - predicted).

        Parameters:
        -----------
        time_index : array-like
            Time indices
        axis_values : array-like
            Observed axis values

        Returns:
        --------
        array : Residuals
        """
        predictions = self.predict(time_index)
        return np.array(axis_values) - predictions

    def get_params(self):
        """
        Get model parameters.

        Returns:
        --------
        dict : Model parameters
        """
        return {
            'axis': self.axis_name,
            'slope': self.slope,
            'intercept': self.intercept,
            'residual_mean': self.residual_mean,
            'residual_std': self.residual_std,
            'is_fitted': self.is_fitted
        }

    def __repr__(self):
        if self.is_fitted:
            return f"AxisRegressionModel({self.axis_name}: y = {self.slope:.6f}*x + {self.intercept:.6f})"
        return f"AxisRegressionModel({self.axis_name}: not fitted)"


class RobotRegressionModels:
    """
    Container for all axis regression models for a robot.

    Manages training, prediction, and visualization for axes 1-12.
    """

    AXIS_NAMES = ['axis_1', 'axis_2', 'axis_3', 'axis_4',
                  'axis_5', 'axis_6', 'axis_7', 'axis_8',
                  'axis_9', 'axis_10', 'axis_11', 'axis_12']

    def __init__(self):
        """Initialize models for all axes."""
        self.models = {axis: AxisRegressionModel(axis) for axis in self.AXIS_NAMES}
        self.training_stats = {}

    def train_all_axes(self, df_training):
        """
        Train regression models for all axes.

        Parameters:
        -----------
        df_training : DataFrame
            Training data with timestamp and axis columns

        Returns:
        --------
        dict : Model parameters for all axes
        """
        print("=" * 60)
        print("TRAINING LINEAR REGRESSION MODELS")
        print("=" * 60)

        # Create time index (sequential integers)
        time_index = np.arange(len(df_training))

        params = {}
        for axis_name in self.AXIS_NAMES:
            if axis_name in df_training.columns:
                print(f"\nTraining model for {axis_name}...")
                axis_values = df_training[axis_name].values

                self.models[axis_name].fit(time_index, axis_values)

                if self.models[axis_name].is_fitted:
                    params[axis_name] = self.models[axis_name].get_params()
                    print(f"  Slope: {params[axis_name]['slope']:.6f}")
                    print(f"  Intercept: {params[axis_name]['intercept']:.6f}")
                    print(f"  Residual Std: {params[axis_name]['residual_std']:.6f}")
                else:
                    print(f"  Skipped (insufficient data)")

        self.training_stats = params
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

        return params

    def predict_all_axes(self, df_test):
        """
        Make predictions for all axes.

        Parameters:
        -----------
        df_test : DataFrame
            Test data with timestamp and axis columns

        Returns:
        --------
        DataFrame : Predictions for each axis
        """
        time_index = np.arange(len(df_test))

        predictions = {}
        for axis_name in self.AXIS_NAMES:
            if self.models[axis_name].is_fitted:
                predictions[f'{axis_name}_predicted'] = self.models[axis_name].predict(time_index)

        return pd.DataFrame(predictions)

    def get_deviations(self, df_test):
        """
        Compute deviations (actual - predicted) for all axes.

        Parameters:
        -----------
        df_test : DataFrame
            Test data

        Returns:
        --------
        DataFrame : Deviations for each axis
        """
        time_index = np.arange(len(df_test))

        deviations = {}
        for axis_name in self.AXIS_NAMES:
            if axis_name in df_test.columns and self.models[axis_name].is_fitted:
                predictions = self.models[axis_name].predict(time_index)
                deviations[f'{axis_name}_deviation'] = df_test[axis_name].values - predictions

        return pd.DataFrame(deviations)

    def plot_regression_lines(self, df, title_prefix="", save_path=None):
        """
        Plot scatter data with regression lines for all axes.

        Parameters:
        -----------
        df : DataFrame
            Data to plot
        title_prefix : str
            Prefix for plot titles
        save_path : str, optional
            Path to save the figure

        Returns:
        --------
        Figure : matplotlib figure
        """
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'{title_prefix}Linear Regression: Time vs Axis Current (All 12 Axes)',
                     fontsize=16, fontweight='bold')

        time_index = np.arange(len(df))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
                  '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2',
                  '#F06292', '#AED581', '#FFD54F', '#90CAF9']

        for idx, axis_name in enumerate(self.AXIS_NAMES):
            row = idx // 4
            col = idx % 4
            ax = axes[row, col]

            if axis_name in df.columns:
                # Check if axis has valid data
                valid_data = df[axis_name].dropna()
                if len(valid_data) > 0:
                    # Scatter plot of actual data
                    ax.scatter(time_index, df[axis_name], alpha=0.3, s=1,
                              color=colors[idx], label='Actual')

                    # Regression line
                    if self.models[axis_name].is_fitted:
                        predictions = self.models[axis_name].predict(time_index)
                        ax.plot(time_index, predictions, color='red', linewidth=2,
                               label=f'Regression (slope={self.models[axis_name].slope:.4f})')

                    ax.set_title(f'{axis_name}', fontsize=12, fontweight='bold')
                    ax.legend(fontsize=8)
                else:
                    ax.set_title(f'{axis_name} (No Data)', fontsize=12, fontweight='bold')
                    ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center',
                           transform=ax.transAxes, fontsize=12, color='gray')
            else:
                ax.set_title(f'{axis_name} (Not Found)', fontsize=12, fontweight='bold')
                ax.text(0.5, 0.5, 'Column Not Found', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, color='gray')

            ax.set_xlabel('Time Index')
            ax.set_ylabel('Current (normalized)')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Regression plot saved to: {save_path}")

        return fig

    def plot_residual_analysis(self, df, save_path=None):
        """
        Plot residual distributions for all axes.

        Parameters:
        -----------
        df : DataFrame
            Data for residual analysis
        save_path : str, optional
            Path to save the figure

        Returns:
        --------
        Figure : matplotlib figure
        """
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('Residual Analysis: Distribution of (Actual - Predicted) - All 12 Axes',
                     fontsize=16, fontweight='bold')

        time_index = np.arange(len(df))

        for idx, axis_name in enumerate(self.AXIS_NAMES):
            row = idx // 4
            col = idx % 4
            ax = axes[row, col]

            if axis_name in df.columns and self.models[axis_name].is_fitted:
                residuals = self.models[axis_name].get_residuals(time_index, df[axis_name])

                # Remove NaN
                residuals = residuals[~np.isnan(residuals)]

                if len(residuals) > 0:
                    # Histogram
                    ax.hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')

                    # Add mean and std lines
                    mean_val = np.mean(residuals)
                    std_val = np.std(residuals)
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                              label=f'Mean: {mean_val:.3f}')
                    ax.axvline(mean_val + 2*std_val, color='orange', linestyle='--',
                              label=f'+2σ: {mean_val + 2*std_val:.3f}')
                    ax.axvline(mean_val - 2*std_val, color='orange', linestyle='--',
                              label=f'-2σ: {mean_val - 2*std_val:.3f}')

                    ax.set_title(f'{axis_name} (σ={std_val:.3f})', fontsize=12, fontweight='bold')
                    ax.legend(fontsize=7)
                else:
                    ax.set_title(f'{axis_name} (No Data)', fontsize=12, fontweight='bold')
                    ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center',
                           transform=ax.transAxes, fontsize=12, color='gray')
            else:
                ax.set_title(f'{axis_name} (No Model)', fontsize=12, fontweight='bold')
                ax.text(0.5, 0.5, 'No Model Fitted', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, color='gray')

            ax.set_xlabel('Residual')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Residual analysis plot saved to: {save_path}")

        return fig

    def plot_residual_boxplots(self, df, save_path=None):
        """
        Plot residual boxplots for outlier identification.

        Parameters:
        -----------
        df : DataFrame
            Data for residual analysis
        save_path : str, optional
            Path to save the figure

        Returns:
        --------
        Figure : matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.suptitle('Residual Boxplots: Outlier Identification',
                     fontsize=16, fontweight='bold')

        time_index = np.arange(len(df))

        residuals_data = []
        labels = []

        for axis_name in self.AXIS_NAMES:
            if axis_name in df.columns and self.models[axis_name].is_fitted:
                residuals = self.models[axis_name].get_residuals(time_index, df[axis_name])
                residuals = residuals[~np.isnan(residuals)]
                residuals_data.append(residuals)
                labels.append(axis_name)

        bp = ax.boxplot(residuals_data, labels=labels, patch_artist=True)

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
                  '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Axis')
        ax.set_ylabel('Residual')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Residual boxplot saved to: {save_path}")

        return fig

    def get_model_summary(self):
        """
        Get summary of all model parameters.

        Returns:
        --------
        DataFrame : Model parameters summary
        """
        summary = []
        for axis_name in self.AXIS_NAMES:
            params = self.models[axis_name].get_params()
            summary.append(params)

        return pd.DataFrame(summary)

    def save_model_params(self, save_path):
        """
        Save model parameters to CSV.

        Parameters:
        -----------
        save_path : str
            Path to save CSV file
        """
        summary = self.get_model_summary()
        summary.to_csv(save_path, index=False)
        print(f"Model parameters saved to: {save_path}")


# Test the module
if __name__ == '__main__':
    print("Testing Linear Regression Models...")

    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    time_index = np.arange(n_samples)

    df_test = pd.DataFrame({
        'axis_1': 0.01 * time_index + np.random.randn(n_samples) * 0.5,
        'axis_2': -0.005 * time_index + np.random.randn(n_samples) * 0.3,
    })

    # Train models
    models = RobotRegressionModels()
    params = models.train_all_axes(df_test)

    print("\nModel Summary:")
    print(models.get_model_summary())

    print("\nLinear Regression Models test complete")
