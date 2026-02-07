# Predictive Maintenance with Linear Regression Alerts

> **Disclaimer**: This project is an extension of the **Data Stream Visualization Workshop**. The following details pertain strictly to the setup and execution of **Predictive Maintenance with Linear Regression-Based Alerts**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Data Generation](#data-generation)
- [Linear Regression for Trend Detection](#linear-regression-for-trend-detection)
- [Residual Analysis and Threshold Discovery](#residual-analysis-and-threshold-discovery)
- [Threshold Justification](#threshold-justification)
- [Alert and Error Module](#alert-and-error-module)
- [Prerequisites](#prerequisites)
- [Installation and Usage](#installation-and-usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)

---

## Overview

This project extends a streaming data pipeline with machine learning models to implement a Predictive Maintenance Alert System for industrial manufacturing robots. The system:

1. **Extends streaming with ML** -- Applies linear regression models on top of a database-backed streaming pipeline to monitor robot current consumption over time.
2. **Detects unusual consumption trends** -- Uses per-axis univariate linear regression to establish baseline behavior, then identifies deviations that signal developing faults.
3. **Analyzes residuals to discover thresholds** -- Computes residuals (observed - predicted), studies their statistical distribution, and derives data-driven thresholds for anomaly detection.
4. **Implements alerts/errors in a streaming context** -- Processes incoming data points sequentially, tracks sustained deviations, and triggers ALERT or ERROR events when thresholds are exceeded for a configurable duration.

---

## Architecture

```
+-----------------------------------------------+
|  RAW Data (RMBR4-2_export_test.csv)           |
|  39K+ robot current measurements              |
+---------------------+-------------------------+
                      |
                      v
+-----------------------------------------------+
|  Data Generation (generate_datasets.py)       |
|  - Outlier removal (2.5 sigma)                |
|  - Z-score normalization                      |
|  - Resample for 4 robots (8-12 axes each)     |
|  - Failure injection (test data only)         |
+-----------+-----------------+-----------------+
            |                 |
            v                 v
  +------------------+  +---------------------+
  | Training Data    |  | Test Data           |
  | 40K rows (clean) |  | 200K rows (failures)|
  +--------+---------+  +----------+----------+
           |                       |
           v                       |
+------------------------------+   |
|  PostgreSQL (Neon.tech)      |   |
|  - Ingest training data      |   |
|  - Query for model training  |   |
+--------------+---------------+   |
               |                   |
               v                   |
+------------------------------+   |
|  Linear Regression Models    |   |
|  - 8-12 axes (per robot)    |   |
|  - slope, intercept, std    |   |
+--------------+---------------+   |
               |                   |
               v                   v
+-----------------------------------------------+
|  Alert System                                 |
|  - Compute deviation per axis                 |
|  - Track sustained anomalies                  |
|  - Trigger ALERT / ERROR events               |
+---------------------+-------------------------+
                      |
                      v
+-----------------------------------------------+
|  Outputs (per robot)                          |
|  - data/model_params.csv (combined w/ robot)  |
|  - logs/alert_log.csv (combined w/ robot)     |
|  - alerts/regression_lines_Robot_*.png (x4)   |
|  - alerts/residual_histograms_Robot_*.png (x4)|
|  - alerts/residual_boxplots_Robot_*.png (x4)  |
|  - alerts/alert_dashboard_Robot_*.png (x4)    |
|  - alerts/alert_dashboard.html (interactive)  |
+-----------------------------------------------+
```

---

## Features

### 1. Database-Backed Training Pipeline
- Training data ingested into Neon.tech PostgreSQL
- Automatic table creation and schema management
- Optimized batch inserts using `psycopg2.execute_values`
- Supports re-ingestion with `FORCE_REINGEST` flag

### 2. Per-Robot, Per-Axis Linear Regression Models
- Separate `RobotRegressionModels` instance trained per robot
- Univariate regression (time index vs. current) for each robot's active axes
- Graceful handling of missing axes via `dropna(axis=1, how='all')`
- Combined model parameters CSV with `robot` column

### 3. Residual-Based Threshold Discovery
- Histogram, boxplot, and statistical summary visualizations
- Data-driven threshold calibration using residual standard deviation
- Per-axis threshold scaling for sensitivity appropriate to each axis

### 4. Per-Robot Streaming Alert Detection
- Separate `AlertSystem` instance per robot
- Test data filtered by robot name and processed through robot's own models
- Per-axis deviation tracking with sustained-duration logic
- Two severity levels: ALERT (early warning) and ERROR (critical)

### 5. Per-Robot Visualization
- Per-robot regression line plots (3x4 grid) saved as `regression_lines_Robot_A.png`, etc.
- Per-robot residual distribution histograms and boxplots
- Per-robot alert dashboard with event markers and duration annotations

### 6. Event Logging and Reporting
- Combined CSV alert log with `robot` column for correct attribution
- Summary statistics grouped by robot, axis, and event type
- Database-ready alert records

### 7. Interactive HTML Dashboard
- Self-contained HTML file generated from `alert_log.csv` via `src/dashboard_generator.py`
- Uses Plotly.js (CDN) for interactive charts -- no additional Python dependencies required
- Summary cards: total warnings, critical alerts, total events, robots monitored
- Robot Health Overview table with status badges (Normal / Needs Attention / Critical) and recommended actions
- Stacked bar charts: anomaly count by robot and most affected axes across all robots
- Deviation timeline scatter plot showing when anomalies occurred with hover details
- Detailed event log with robot/type filtering and column sorting
- Dark-themed responsive layout designed for Plant Maintenance Engineers
- Auto-opens in default browser after generation

---

## Data Generation

Both training and test datasets are generated programmatically from raw robot sensor data using `data/generate_datasets.py`.

### Source Data

`RMBR4-2_export_test.csv` contains ~39,000 raw current measurements (0-37A) from Robot A's controller across 8 axes. This data includes normal operation as well as potential failure points.

### Generation Process

```
RAW Data (RMBR4-2_export_test.csv)
        |
        v
   Filter idle rows (all-zero readings)
        |
        v
   Remove outliers (> 2.5 sigma from active-row mean)
        |
        v
   Z-score normalization (per-axis mean=0, std=1)
        |
        v
   Resample with replacement + per-robot offset/jitter
        |
        +---> Training Data (robots_combined_traindata.csv)
        |         Clean data, no failure patterns
        |
        +---> Test Data (robots_combined_v2.csv)
                  Clean base + gradual drift (ALERT) + brief spikes (ERROR)
```

### Training Data (`robots_combined_traindata.csv`)

- **Purpose**: Establish baseline behavior for each robot axis
- **Content**: Clean, normalized data with outliers removed -- no failure patterns
- **Size**: 40,000 rows (10,000 per robot)
- Robots A, B, C, D are extrapolated from Robot A's clean data with small per-robot offsets to simulate fleet variation

### Test Data (`robots_combined_v2.csv`)

- **Purpose**: Evaluate the alert system's ability to detect developing failures
- **Content**: Mostly clean data with subtle failure injection in the final ~10%
- **Size**: 200,000 rows (50,000 per robot)
- **Failure design**: Gradual sinusoidal drift (2.3-2.6 sigma) simulating degradation developing over ~2 weeks, with brief bell-shaped spikes (~250 points) just above 3 sigma
- Each robot degrades on different axes to demonstrate varied failure modes
- Expected outcome: mostly ALERT events with only 1-2 ERROR events per robot

### Robot Axis Configuration

| Robot   | Active Axes                | Description                                              |
|---------|----------------------------|----------------------------------------------------------|
| Robot A | axis_1 -- axis_8 (8 axes)  | Matches RAW data axis count                              |
| Robot B | axis_1 -- axis_10 (10 axes)| Axes 9-10 synthesized from weighted combinations of 1-8  |
| Robot C | axis_1 -- axis_12 (12 axes)| Axes 9-12 synthesized from weighted combinations of 1-8  |
| Robot D | axis_1 -- axis_12 (12 axes)| Axes 9-12 synthesized from weighted combinations of 1-8  |

### Regenerating Data

To regenerate both CSVs from the raw source:

```bash
cd data
python generate_datasets.py
```

After regenerating, set `FORCE_REINGEST = True` in the notebook (Cell 2) to reload the database.

---

## Linear Regression for Trend Detection

For each axis (1-12), a univariate linear regression model is trained on the clean training data:

```
y = slope * time_index + intercept
```

Where:
- `y` = predicted current value (normalized)
- `time_index` = sequential integer representing time
- `slope` = rate of change over time
- `intercept` = baseline current value

The model captures the expected consumption trend for each axis. During testing, each incoming data point is compared against the regression prediction. The difference (residual) reveals whether the robot is operating normally or deviating from its baseline.

Models are trained only on axes that contain data for a given robot. Axes without data (e.g., axes 9-12 for Robot A) are skipped gracefully.

---

## Residual Analysis and Threshold Discovery

After training, residuals are computed on the training data:

```
residual = actual_value - predicted_value
```

The system then analyzes the residual distribution for each axis through:

1. **Histograms** -- Visualize the shape of residual distributions (approximately normal for healthy data)
2. **Boxplots** -- Identify outlier boundaries and interquartile spread
3. **Statistical summary** -- Mean, std, percentiles (Q1, median, Q3, P95, P99) per axis

These analyses reveal that healthy operation produces residuals concentrated within +/-2 standard deviations. Values beyond this range are rare under normal conditions, making them reliable indicators of anomalous behavior. The per-axis `residual_std` becomes the scaling factor for alert thresholds.

---

## Threshold Justification

### MinC = 3.5 (Alert Threshold)

**Definition**: Minimum deviation (in units of residual std) above regression line to trigger ALERT.

**Justification**:
- Values exceeding 3.5 std from the regression line are very rare under normal operating conditions (less than **1 in 4,300 chance**)
- This threshold effectively eliminates false positives from normal process variation while reliably capturing genuine degradation onset
- In predictive maintenance context, this indicates **early signs of potential degradation** that warrant monitoring and scheduled inspection

**Evidence**: Residual histogram and boxplot analyses show normal distribution concentrated within +/-2 std. Values beyond 3.5 std are exceedingly rare during healthy operation, making them reliable indicators of anomalous behavior.

### MaxC = 4.0 (Error Threshold)

**Definition**: Maximum deviation (in units of residual std) above regression line to trigger ERROR.

**Justification**:
- Values exceeding 4 std are extremely rare under normal conditions (less than **1 in 15,787 chance**)
- This level represents an unambiguously anomalous deviation that signals **critical failure in progress**
- Requires immediate intervention to prevent catastrophic breakdown

**Evidence**: Boxplot analysis shows values beyond 4 std align with outlier patterns that historically precede equipment failures. At this level, the deviation is far outside any normal operational variance.

### T = 5 seconds (Sustained Duration)

**Definition**: Minimum continuous time that deviation must persist to trigger an alert.

**Justification**:
- **Transient spikes are normal** during robot operation cycles (start/stop, load changes)
- A 5-second sustained deviation window is calibrated to the high σ thresholds used -- since deviations at 3.5-4σ are already extremely unlikely under normal conditions, a shorter sustain window is sufficient to confirm genuine anomalies
- **Filters out single-point transient spikes** while capturing persistent issues
- Prevents false positives from momentary operational transients

**Evidence**: Time series analysis shows healthy operation includes brief spikes that resolve within 1-3 seconds. Sustained deviations at 3.5σ+ for 5 consecutive seconds reliably indicate genuine anomalies.

---

## Alert and Error Module

The alert system (`src/alert_system.py`) processes data sequentially, simulating a streaming context. For each incoming data point, it:

1. Computes the deviation from the regression prediction
2. Classifies the deviation level using axis-specific thresholds scaled by `residual_std`
3. Tracks sustained deviations using per-axis state trackers
4. Triggers events when duration exceeds threshold `T`

### Alert Rules

| Level      | Condition                              | Action                       |
|------------|----------------------------------------|------------------------------|
| **NORMAL** | abs(deviation) < MinC * residual_std   | Continue monitoring          |
| **ALERT**  | abs(deviation) >= MinC * residual_std for >= T seconds | Schedule inspection |
| **ERROR**  | abs(deviation) >= MaxC * residual_std for >= T seconds | Immediate attention required |

### Event Logging

All triggered events are logged with: timestamp, axis, event type, deviation magnitude, duration, actual vs. predicted values, threshold used, and robot name. Logs are saved to `logs/alert_log.csv`.

### Visualization

Per-robot 3x4 dashboards (`alerts/alert_dashboard_Robot_A.png`, etc.) display each robot's axes with ALERT/ERROR markers overlaid on the time series, providing a clear view of individual robot health.

An interactive HTML dashboard (`alerts/alert_dashboard.html`) is also generated from the combined alert log using `src/dashboard_generator.py`. It provides summary cards, a robot health overview table, stacked bar charts (by robot and axis), a deviation timeline, and a filterable/sortable event log -- all powered by Plotly.js (CDN) with no extra Python dependencies. The dashboard auto-opens in the default browser after generation.

---

## Prerequisites

Before running `PredictiveMaintenance_LinReg.ipynb`, ensure the following are in place:

### Software Requirements

- **Python 3.8+**
- **Jupyter Notebook** (`jupyter`, `notebook`, `ipykernel`)
- **PyQt5** (required for `%matplotlib qt` backend used by the notebook)

### Python Dependencies

All listed in `requirements.txt`:

| Package          | Purpose                                   |
|------------------|-------------------------------------------|
| pandas >= 2.0    | Data manipulation                         |
| numpy >= 1.24    | Numerical computation                     |
| matplotlib >= 3.7| Visualization and plotting                |
| seaborn >= 0.12  | Statistical visualization                 |
| scikit-learn >= 1.3 | Linear regression models               |
| psycopg2-binary >= 2.9 | PostgreSQL database adapter          |
| PyQt5 >= 5.15    | Matplotlib Qt backend for plot windows    |

### Database

- **Neon.tech PostgreSQL** account (free tier available at [neon.tech](https://neon.tech/))
- Connection details configured in Cell 2 of the notebook (`db_config`)

### Data Files

The following must exist in the `data/` directory:

| File | Required | Notes |
|------|----------|-------|
| `RMBR4-2_export_test.csv` | Only if regenerating data | RAW source |
| `robots_combined_traindata.csv` | Yes | Training data (pre-generated or run `generate_datasets.py`) |
| `robots_combined_v2.csv` | Yes | Test data (pre-generated or run `generate_datasets.py`) |

---

## Installation and Usage

### Step 1: Clone and Set Up Environment

```bash
git clone https://github.com/muthuacumen/Lab1_StreamingDataforPM-with-Linear-Regression-Alerts.git
cd Lab1_StreamingDataforPMwithLinRegAlerts

python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Configure Database

1. Sign up at [Neon.tech](https://neon.tech/) and create a project
2. Update `db_config` in the notebook (Cell 2) with your connection details

### Step 3: (Optional) Regenerate Data

Skip this step if `robots_combined_traindata.csv` and `robots_combined_v2.csv` already exist.

```bash
cd data
python generate_datasets.py
```

### Step 4: Run the Notebook

```bash
cd notebook
jupyter notebook PredictiveMaintenance_LinReg.ipynb
```

Execute all cells in order. The notebook will:

1. Connect to the database and ingest training data
2. Train linear regression models for each robot's active axes
3. Visualize per-robot regression lines (scatter plots with fitted lines)
4. Compute residuals and visualize distributions (histograms, boxplots)
5. Configure thresholds with justification
6. Process all test data through the alert system
7. Log events to CSV and generate per-robot alert dashboards (PNG)
8. Generate interactive HTML dashboard and open it in the default browser

### Output Files

| File | Location | Description |
|------|----------|-------------|
| `model_params.csv` | `data/` | Combined model parameters with `robot` column |
| `alert_log.csv` | `logs/` | Combined ALERT/ERROR events for all robots with `robot` column |
| `regression_lines_Robot_A.png` (x4) | `alerts/` | Per-robot regression line plots (3x4 grid) |
| `residual_histograms_Robot_A.png` (x4) | `alerts/` | Per-robot residual distribution analysis |
| `residual_boxplots_Robot_A.png` (x4) | `alerts/` | Per-robot outlier identification boxplots |
| `alert_dashboard_Robot_A.png` (x4) | `alerts/` | Per-robot alert dashboard (3x4 grid) |
| `alert_dashboard.html` | `alerts/` | Interactive HTML dashboard (Plotly.js, auto-opens in browser) |

---

## Project Structure

```
Lab1_StreamingDataforPMwithLinRegAlerts/
|
|-- README.md                                    # This file
|-- requirements.txt                             # Python dependencies
|-- claude_implementation.md                     # Implementation guide
|
|-- src/                                         # Source code
|   |-- database_utils.py                        # Database connection and ingestion
|   |-- linear_regression_model.py               # Per-axis linear regression models
|   |-- alert_system.py                          # Alert/Error detection and logging
|   |-- dashboard_generator.py                   # Interactive HTML dashboard generator
|   |-- StreamingSimulator.py                    # Original workshop streaming class
|
|-- notebook/                                    # Jupyter notebooks
|   |-- PredictiveMaintenance_LinReg.ipynb       # Main PM notebook (run this)
|   |-- DataStreamVisualization_workshop.ipynb   # Original streaming workshop
|
|-- data/                                        # Data directory
|   |-- RMBR4-2_export_test.csv                  # RAW robot data (39K+ rows)
|   |-- generate_datasets.py                     # Data generation script
|   |-- robots_combined_traindata.csv            # Training data (40K rows, clean)
|   |-- robots_combined_v2.csv                   # Test data (200K rows, with failures)
|   |-- normalization_params.csv                 # Z-score normalization parameters
|   |-- model_params.csv                         # Trained model parameters (generated)
|
|-- logs/                                        # Log files
|   |-- alert_log.csv                            # Combined alert/error log (all robots)
|
|-- alerts/                                      # Alert visualizations (per robot)
|   |-- regression_lines_Robot_A.png             # Per-robot regression plots (x4)
|   |-- residual_histograms_Robot_A.png          # Per-robot residual analysis (x4)
|   |-- residual_boxplots_Robot_A.png            # Per-robot outlier detection (x4)
|   |-- alert_dashboard_Robot_A.png              # Per-robot alert dashboard (x4)
|   |-- alert_dashboard.html                     # Interactive HTML dashboard (generated)
|
|-- .venv/                                       # Virtual environment (not tracked)
```

---

## Technologies Used

### Core

| Technology | Purpose |
|------------|---------|
| Python 3.8+ | Primary programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computing |
| scikit-learn | Linear regression model training |
| Matplotlib | Visualization and dashboard generation |
| Seaborn | Statistical visualization (residual analysis) |

### Database

| Technology | Purpose |
|------------|---------|
| PostgreSQL | Relational database for training data persistence |
| psycopg2 | PostgreSQL adapter with `execute_values` for batch inserts |
| Neon.tech | Serverless PostgreSQL hosting |

### Interactive Dashboard

| Technology | Purpose |
|------------|---------|
| Plotly.js (CDN) | Interactive charts in the HTML dashboard (no Python install needed) |

### Environment

| Technology | Purpose |
|------------|---------|
| Jupyter Notebook | Interactive execution environment |
| PyQt5 | Matplotlib Qt backend for plot windows |

---

## License

This project is for educational purposes. If using real manufacturing data, ensure compliance with company data policies and privacy regulations.
