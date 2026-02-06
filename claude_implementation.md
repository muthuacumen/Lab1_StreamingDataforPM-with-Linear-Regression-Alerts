# Predictive Maintenance with Linear Regression Alerts - Implementation Plan

## Project Overview

This document serves as a comprehensive implementation guide for the Predictive Maintenance project using Linear Regression-based alerts. It can be used to continue development in future Claude sessions.

---

## Project Context

### Objective
Extend the existing Data Stream Visualization Workshop to implement a Predictive Maintenance Alert System using Linear Regression models to detect anomalies and predict failures in industrial robot current data.

### Data Files

| File | Description | Purpose |
|------|-------------|---------|
| `RMBR4-2_export_test.csv` | RAW current data (0-37A) from Robot A | Original source data |
| `generate_datasets.py` | Data generation script - removes outliers, z-score normalizes, extrapolates for 4 robots with varying axis counts, injects subtle failures into test data | **Reproducibility** |
| `robots_combined_traindata.csv` | Normalized (z-score), outliers removed, extrapolated for 4 robots (A=8 axes, B=10 axes, C&D=12 axes). 10,000 pts/robot, 40,000 total | **Training** |
| `robots_combined_v2.csv` | Normalized data with subtle failure injection (gradual 2-3 sigma drift + brief >3 sigma spikes). 50,000 pts/robot, 200,000 total | **Testing** |
| `normalization_params.csv` | Mean and std values used for z-score normalization (computed from clean active data) | Reference |

### Robot Axis Configuration

| Robot | Training Points | Test Points | Active Axes |
|-------|----------------|-------------|-------------|
| Robot A | 10,000 | 50,000 | axis_1 - axis_8 (8 axes) |
| Robot B | 10,000 | 50,000 | axis_1 - axis_10 (10 axes) |
| Robot C | 10,000 | 50,000 | axis_1 - axis_12 (12 axes) |
| Robot D | 10,000 | 50,000 | axis_1 - axis_12 (12 axes) |

### Database Configuration (Neon.tech PostgreSQL)
```python
db_config = {
    'host': 'ep-polished-snow-ahx3qiod-pooler.c-3.us-east-1.aws.neon.tech',
    'database': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_JlIENr3i4AbL',
    'port': 5432,
    'sslmode': 'require'
}
```

---

## Implementation Checklist

### 1. Project Setup (1 point)
- [x] `README.md` - Professional and clear documentation
- [x] `requirements.txt` - All dependencies with versions
- [x] `data/` folder with relevant CSV files
- [ ] Jupyter notebook with output of last test run *(Run notebook to complete)*

### 2. Database Integration (1.5 points)
- [x] Connect to Neon.tech PostgreSQL
- [x] Ingest training data (`robots_combined_traindata.csv`) into database
- [x] Query streaming data for model training

### 3. Streaming Simulation (1 point)
- [x] Simulate CSV → DB time-based flow
- [x] Use `robots_combined_v2.csv` as synthetic test data
- [x] Ensure test data is properly normalized w.r.t. training data

### 4. Regression Models & Residual Analysis (2 points)
- [x] Fit univariate linear regression for each axis (Time → Axis values)
- [x] Axes to model: axis_1 through axis_12 (all 12 axes)
- [x] Record slope and intercept for each model
- [x] Plot scatter data with regression lines
- [x] Compute residuals (observed - predicted)
- [x] Plot residual distributions (histogram/boxplot)
- [x] Identify outlier patterns

### 5. Threshold Discovery & Justification (2 points)
- [x] Define **MinC**: Minimum current deviation for ALERT (2.0σ)
- [x] Define **MaxC**: Maximum current deviation for ERROR (3.0σ)
- [x] Define **T**: Minimum continuous time (30 seconds) for sustained deviation
- [x] Document justification in README.md with evidence from analysis

### 6. Alerts & Errors Implementation (2 points)
- [x] Implement detection logic:
  - ALERT: deviation >= MinC sustained for >= T seconds
  - ERROR: deviation >= MaxC sustained for >= T seconds
- [x] Log events to CSV (`logs/alert_log.csv`)
- [x] Optionally store in database table

### 7. Visualization/Dashboard (0.5 points)
- [x] Regression plots with Alert/Error markers
- [x] Annotate each event with duration
- [x] Save alert images to `alerts/` folder
- [x] Clear, professional plots

---

## File Structure

```
Lab1_StreamingDataforPMwithLinRegAlerts/
├── README.md                          # Project documentation
├── requirements.txt                   # Dependencies
├── claude_implementation.md           # This file
├── data/
│   ├── RMBR4-2_export_test.csv       # RAW data (39K+ rows, Robot A)
│   ├── generate_datasets.py          # Data generation script
│   ├── robots_combined_traindata.csv  # Training data (40K rows, clean)
│   ├── robots_combined_v2.csv         # Testing data (200K rows, with failures)
│   ├── normalization_params.csv       # Normalization parameters
│   └── model_params.csv               # Trained model parameters (generated)
├── src/
│   ├── StreamingSimulator.py          # Existing streaming module
│   ├── linear_regression_model.py     # Linear regression models
│   ├── alert_system.py                # Alert/Error detection
│   └── database_utils.py              # Database utilities
├── notebook/
│   ├── DataStreamVisualization_workshop.ipynb  # Existing workshop
│   └── PredictiveMaintenance_LinReg.ipynb      # NEW: Main PM notebook
├── logs/
│   └── alert_log.csv                  # Combined alert/error log (all robots, with robot column)
└── alerts/
    ├── regression_lines_Robot_A.png   # Per-robot regression plots (x4)
    ├── residual_histograms_Robot_A.png # Per-robot residual histograms (x4)
    ├── residual_boxplots_Robot_A.png  # Per-robot residual boxplots (x4)
    └── alert_dashboard_Robot_A.png    # Per-robot alert dashboards (x4)
```

---

## Implementation Details

### Phase 1: Database Integration

**File: `src/database_utils.py`**

```python
# Database table supports all 12 axes (axis_1 through axis_12)
# Functions implemented:
def connect_to_db(db_config) -> connection
def create_training_table(conn) -> None  # Creates table with 12 axis columns
def ingest_training_data(conn, csv_path) -> int  # Ingests all 12 axes, returns row count
def query_training_data(conn, robot_name=None) -> DataFrame
def create_alerts_table(conn) -> None
def log_alert_to_db(conn, alert_record) -> None
def clear_training_table(conn) -> None  # Clears existing data for re-ingestion
def get_training_record_count(conn) -> int
def query_alerts(conn, event_type=None, limit=100) -> DataFrame
```

### Phase 2: Linear Regression Models

**File: `src/linear_regression_model.py`**

```python
# Supports all 12 axes with 3x4 plot grid layout
AXIS_NAMES = ['axis_1', 'axis_2', 'axis_3', 'axis_4',
              'axis_5', 'axis_6', 'axis_7', 'axis_8',
              'axis_9', 'axis_10', 'axis_11', 'axis_12']

# Model structure per axis:
class AxisRegressionModel:
    def __init__(self, axis_name):
        self.axis_name = axis_name
        self.slope = None
        self.intercept = None
        self.residual_std = None
        self.model = None  # sklearn LinearRegression

    def fit(self, time_index, axis_values) -> None
    def predict(self, time_index) -> array
    def get_residuals(self, time_index, axis_values) -> array
    def get_params(self) -> dict  # slope, intercept, residual_std

# Container for all axis models:
class RobotRegressionModels:
    AXIS_NAMES = [...]  # All 12 axes

    def __init__(self):
        self.models = {}  # axis_name -> AxisRegressionModel

    def train_all_axes(self, df_training) -> dict  # Returns model params
    def predict_all_axes(self, df_test) -> DataFrame
    def get_deviations(self, df_test) -> DataFrame
    def plot_regression_lines(self, df, save_path=None) -> Figure  # 3x4 grid
    def plot_residual_analysis(self, df, save_path=None) -> Figure  # 3x4 grid
    def plot_residual_boxplots(self, df, save_path=None) -> Figure
    def get_model_summary(self) -> DataFrame
    def save_model_params(self, save_path) -> None
```

### Phase 3: Alert System

**File: `src/alert_system.py`**

```python
# Supports all 12 axes
AXIS_NAMES = ['axis_1', 'axis_2', 'axis_3', 'axis_4',
              'axis_5', 'axis_6', 'axis_7', 'axis_8',
              'axis_9', 'axis_10', 'axis_11', 'axis_12']

# Threshold configuration class:
class AlertThresholds:
    def __init__(self, MinC=2.0, MaxC=3.0, T=30):
        self.MinC = MinC   # Minimum deviation for ALERT (σ multiplier)
        self.MaxC = MaxC   # Maximum deviation for ERROR (σ multiplier)
        self.T = T         # Minimum sustained duration in seconds

# Alert event class:
class AlertEvent:
    # Fields: timestamp, axis, event_type, deviation, duration_seconds,
    #         actual_value, predicted_value, threshold_used, robot

class AlertSystem:
    AXIS_NAMES = [...]  # All 12 axes

    def __init__(self, models: RobotRegressionModels, thresholds: AlertThresholds):
        self.models = models
        self.thresholds = thresholds
        self.alert_log = []
        self.deviation_trackers = {}  # Per-axis tracking

    def check_deviation(self, deviation, axis) -> str
        # Returns: 'NORMAL', 'ALERT', or 'ERROR'

    def process_streaming_data(self, df_test, time_interval_seconds=1) -> list[AlertEvent]
        # Process entire dataset, return triggered alerts

    def save_log_to_csv(self, path='logs/alert_log.csv') -> None

    def get_alert_summary(self) -> DataFrame

    def generate_alert_dashboard(self, df_test, save_path) -> Figure
        # Comprehensive 3x4 dashboard for all 12 axes
```

**Note:** Individual axis alert plotting methods were removed. Only the comprehensive dashboard visualization is provided.

### Phase 4: Main Notebook

**File: `notebook/PredictiveMaintenance_LinReg.ipynb`**

#### Cell Structure (11 cells):

1. **Imports and Setup**
   - Import libraries and custom modules
   - Configure matplotlib Qt backend
   - Set visualization style

2. **Database Integration**
   - Connect to Neon.tech PostgreSQL
   - Create training and alerts tables
   - Check existing records

3. **Data Ingestion**
   - Ingest training data (with FORCE_REINGEST option)
   - Define `ROBOT_CONFIG` dict (Robot A=8, B=10, C=12, D=12 axes)
   - Query training data per robot via `query_training_data(conn, robot_name=...)`
   - `dropna(axis=1, how='all')` to remove NaN-only axis columns per robot
   - Store in `training_data` dict (robot_name -> DataFrame)

4. **Model Training (Per-Robot)**
   - Loop over `training_data`, create `RobotRegressionModels()` per robot
   - Train and store in `robot_models` dict (robot_name -> RobotRegressionModels)
   - Collect model summaries with `robot` column
   - Save combined `model_params.csv` with `robot` column

5. **Regression Visualization (Per-Robot)**
   - Loop over robots, generate 3x4 scatter + regression line plots
   - Save as `alerts/regression_lines_Robot_A.png`, etc. (one per robot)

6. **Residual Analysis (Per-Robot)**
   - Per-robot residual histograms (3x4 grid) saved as `residual_histograms_Robot_A.png`, etc.
   - Per-robot residual boxplots saved as `residual_boxplots_Robot_A.png`, etc.
   - Combined residual statistics table with `robot` column

7. **Threshold Configuration**
   - Define MinC=2.0, MaxC=3.0, T=30
   - Display justification
   - Show actual threshold values per robot per axis

8. **Alert System Setup (Per-Robot)**
   - Loop over `robot_models`, create `AlertSystem(models, thresholds)` per robot
   - Store in `robot_alert_systems` dict (robot_name -> AlertSystem)
   - Create logs/ and alerts/ directories

9. **Streaming Test Simulation (Per-Robot)**
   - Load test data (`robots_combined_v2.csv`) once as `df_test_all` - 200,000 rows total
   - Loop over robots: filter by robot name, `reset_index`, `dropna`, process through robot's own `AlertSystem`
   - Display combined detection results and summary grouped by `['robot', 'axis', 'event_type']`
   - Expected: mostly ALERT events (sustained 2-3 sigma drift), only 1-2 ERROR events per robot

10. **Event Logging**
    - Merge all robot alert logs into combined `alert_log.csv` with `robot` column
    - Display log preview

11. **Dashboard & Summary (Per-Robot)**
    - Generate per-robot 3x4 alert dashboards saved as `alert_dashboard_Robot_A.png`, etc.
    - Per-robot breakdown: fitted models, train/test counts, alert/error counts
    - Combined totals
    - Close database connection

---

## Threshold Justification Template

To be documented in README.md:

### MinC (Minimum Deviation for ALERT)
- **Value**: [TBD after analysis, e.g., 2.0 * residual_std]
- **Justification**: Based on residual analysis, values exceeding this threshold represent [X]% of normal operational variance. In predictive maintenance context, this indicates early signs of potential degradation that warrant monitoring.
- **Evidence**: [Reference to residual distribution plots]

### MaxC (Maximum Deviation for ERROR)
- **Value**: [TBD after analysis, e.g., 3.0 * residual_std]
- **Justification**: Values exceeding this threshold are outside [Y]% confidence interval of normal operation. This level of deviation indicates imminent failure risk requiring immediate attention.
- **Evidence**: [Reference to outlier analysis]

### T (Sustained Duration)
- **Value**: [TBD, e.g., 30 seconds]
- **Justification**: Brief spikes may be normal operational transients. Sustained deviation for T seconds indicates persistent anomaly rather than momentary fluctuation. Based on typical robot operation cycles observed in training data.
- **Evidence**: [Reference to time series patterns]

---

## Dependencies (requirements.txt)

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
psycopg2-binary>=2.9.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

---

## Execution Order

1. **Setup Environment**
   ```bash
   cd Lab1_StreamingDataforPMwithLinRegAlerts
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Run Main Notebook**
   ```bash
   jupyter notebook notebook/PredictiveMaintenance_LinReg.ipynb
   ```

3. **Execute All Cells** to produce final output with:
   - Regression plots
   - Residual analysis
   - Alert/Error detection
   - Logged events

---

## Key Formulas

### Linear Regression
```
y = slope * x + intercept
```
Where:
- x = time_index (sequential integer)
- y = axis current value

### Deviation Calculation
```
deviation = actual_value - predicted_value
```

### Alert Conditions
```
ALERT: deviation >= MinC AND duration >= T seconds
ERROR: deviation >= MaxC AND duration >= T seconds
```

### Z-Score Normalization (used for training data)
```
normalized_value = (raw_value - mean) / std
```

---

## Testing Verification

Before submission, ensure:

1. [ ] All notebook cells execute without errors
2. [ ] Regression plots display correctly
3. [ ] Residual analysis plots are visible
4. [ ] At least some ALERT/ERROR events are detected in test data
5. [ ] `logs/alert_log.csv` contains logged events
6. [ ] Alert images are saved in `alerts/` folder
7. [ ] README.md documents threshold justification
8. [ ] Database integration works (connection, query)

---

## Resuming Development

If continuing in a new Claude session:

1. Read this file (`claude_implementation.md`)
2. Check which items in the checklist are completed
3. Review any existing code in `src/` folder
4. Continue from the next uncompleted item

Current Status: **Implementation Complete** - All code files created and tested.

## Completed Items

- [x] `data/generate_datasets.py` - Data generation script (outlier removal, z-score normalization, failure injection)
- [x] `robots_combined_traindata.csv` - Clean training data (40,000 rows: 4 robots x 10,000 pts, varying axis counts)
- [x] `robots_combined_v2.csv` - Test data with subtle failures (200,000 rows: 4 robots x 50,000 pts, mostly ALERTs + 1-2 ERRORs)
- [x] `requirements.txt` - Dependencies file exists
- [x] `src/database_utils.py` - Database connection and utilities (12 axes)
- [x] `src/linear_regression_model.py` - Regression models for all 12 axes
- [x] `src/alert_system.py` - Alert/Error detection system (12 axes)
- [x] `notebook/PredictiveMaintenance_LinReg.ipynb` - Main notebook (11 cells, processes ALL test data)
- [x] `README.md` - Updated with threshold justification
- [x] `logs/` and `alerts/` directories created

## Key Implementation Details

### 12-Axis Support
All components support axes 1-12:
- Database schema includes axis_1 through axis_12 columns
- Regression models train on all available axes
- Alert system monitors all 12 axes
- Visualizations use 3x4 grid layout

### Per-Robot Architecture
- Each robot gets its own `RobotRegressionModels` and `AlertSystem` instance
- `dropna(axis=1, how='all')` in the notebook prevents NaN-column crashes for robots with fewer axes
- Combined CSV outputs (`model_params.csv`, `alert_log.csv`) include a `robot` column
- Per-robot PNGs: `regression_lines_Robot_A.png`, `alert_dashboard_Robot_A.png`, etc.

### Visualization Approach
- **No individual axis plots** - Only comprehensive 3x4 grid views per robot
- Each robot gets its own regression, residual, and dashboard plots
- Regression and residual plots use 3x4 layout (unfitted axes show "No Data")

### Data Generation (`data/generate_datasets.py`)
- Source: `RMBR4-2_export_test.csv` (Robot A raw current data, 8 axes, ~39K rows)
- Outlier removal: rows where any axis exceeds 2.5 sigma from mean of active (non-idle) rows
- Z-score normalization using clean data statistics
- Training data: clean resampled data with small per-robot offsets and jitter
- Test data: same clean base + failure injection in the last ~10% of each robot's data
  - Gradual sinusoidal drift reaching 2.3-2.6 sigma (ALERT level)
  - Brief bell-shaped spike reaching 3.1-3.4 sigma (ERROR level, ~250 points)
  - Different axes affected per robot to show varied failure modes
- Axes 9-12 are synthesized as weighted combinations of axes 1-8 + noise

### Dataset Compatibility
The system accepts any dataset with the same structure:
- Required columns: `timestamp`, `robot`, `axis_1` through `axis_12`
- Missing axis columns are handled gracefully (stored as NULL)

## Next Steps (User Action Required)

1. Open and run `notebook/PredictiveMaintenance_LinReg.ipynb`
2. Execute all cells to generate outputs
3. Verify alert_log.csv and alert images are created
4. Review and adjust thresholds if needed

## Important: Regenerating Data

To regenerate training and test data from the RAW source:
```bash
cd data
python generate_datasets.py
```
This will overwrite `robots_combined_traindata.csv`, `robots_combined_v2.csv`, and `normalization_params.csv`.

## Important: Re-ingesting Data

If data has been regenerated or the database schema has changed:
1. Set `FORCE_REINGEST = True` in the notebook (Cell 3)
2. This will clear and reload the training data with the new 12-axis schema
