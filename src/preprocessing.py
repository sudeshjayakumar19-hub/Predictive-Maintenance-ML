"""
NASA C-MAPSS FD001 Preprocessing and Feature Engineering Module
Predictive Maintenance ML Capstone Project

This module prepares the raw C-MAPSS FD001 training dataset by:
1. Loading and applying standardized domain column names.
2. Chronologically sorting by engine ID and operational cycle.
3. Calculating ground truth Remaining Useful Life (RUL) for each cycle observation.
4. Defining a binary failure-risk target (FAILURE_WINDOW = 30 cycles).
5. Extracting strictly causal, time-aware lag and rolling statistical features
   (using only historical/current observations to avoid target leakage).
6. Validating data integrity, saving processed datasets, and generating reports/plots.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Global constants
FAILURE_WINDOW = 30

INDEX_COLUMNS = ['unit_number', 'time_in_cycles']
SETTING_COLUMNS = ['op_setting_1', 'op_setting_2', 'op_setting_3']
SENSOR_COLUMNS = [f'sensor_{i}' for i in range(1, 22)]
ALL_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

# Informative sensors (active degradation indicators with non-zero variance in FD001)
INFORMATIVE_SENSORS = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8',
    'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
    'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'
]

# Constant/near-constant sensors in FD001 operating conditions
CONSTANT_SENSORS = [
    'sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16',
    'sensor_18', 'sensor_19'
]


def load_dataset(filepath: str = 'data/train_FD001.txt') -> pd.DataFrame:
    """
    Load raw space-delimited C-MAPSS FD001 file and assign standard column headers.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=ALL_COLUMNS)
    return df


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure rows are strictly sorted chronologically by unit_number and time_in_cycles.
    """
    sorted_df = df.sort_values(by=['unit_number', 'time_in_cycles'], ascending=[True, True]).reset_index(drop=True)
    return sorted_df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verify and handle any missing or corrupt values.
    """
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        # Groupwise forward fill then backward fill if missing data is encountered
        df = df.groupby('unit_number').apply(lambda group: group.ffill().bfill()).reset_index(drop=True)
    return df


def calculate_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Remaining Useful Life (RUL) for each cycle observation.
    For each engine i: RUL_t = max(cycle_i) - cycle_{i, t}
    """
    max_cycles = df.groupby('unit_number')['time_in_cycles'].transform('max')
    df['RUL'] = max_cycles - df['time_in_cycles']
    return df


def create_failure_target(df: pd.DataFrame, failure_window: int = FAILURE_WINDOW) -> pd.DataFrame:
    """
    Create binary failure-risk target indicator based on prediction window.
    target = 1 if RUL <= failure_window else 0
    """
    df['target'] = (df['RUL'] <= failure_window).astype(int)
    return df


def create_engineered_features(
    df: pd.DataFrame,
    sensors: list = None,
    rolling_windows: list = [5]
) -> pd.DataFrame:
    """
    Generate time-aware lag and rolling statistical features per engine.
    
    Guarantees:
    - Grouped strictly per engine (`unit_number`) to prevent data leaking across engines.
    - Uses only past and current records (strictly backward-looking rolling and positive lag).
    - Preserves exact index alignment.
    
    Features created per sensor:
    - lag_1: observation at t - 1 (initial observation t=1 is filled with current value)
    - delta_1: change from previous observation (t) - (t - 1)
    - rolling_mean_{w}: backward rolling mean over window w (min_periods=1)
    - rolling_std_{w}: backward rolling standard deviation over window w (filled with 0 for single sample)
    """
    if sensors is None:
        sensors = INFORMATIVE_SENSORS

    engineered_df = df.copy()

    for sensor in sensors:
        # Lag-1 per engine
        lag_series = engineered_df.groupby('unit_number')[sensor].shift(1)
        engineered_df[f'{sensor}_lag1'] = lag_series.fillna(engineered_df[sensor])
        
        # Delta / Change per engine
        diff_series = engineered_df.groupby('unit_number')[sensor].diff(1)
        engineered_df[f'{sensor}_delta'] = diff_series.fillna(0.0)

        # Rolling statistics per engine
        for w in rolling_windows:
            # Pandas rolling grouped by unit_number
            rolling_obj = engineered_df.groupby('unit_number')[sensor].rolling(window=w, min_periods=1)
            
            # Extract rolling mean & std matching index
            r_mean = rolling_obj.mean().reset_index(level=0, drop=True)
            r_std = rolling_obj.std().reset_index(level=0, drop=True).fillna(0.0)

            engineered_df[f'{sensor}_roll_mean_{w}'] = r_mean
            engineered_df[f'{sensor}_roll_std_{w}'] = r_std

    return engineered_df


def generate_data_summary(df: pd.DataFrame, output_path: str = 'outputs/data_summary.txt'):
    """
    Compute key dataset summary metrics and save to outputs/data_summary.txt.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_rows, n_cols = df.shape
    n_engines = df['unit_number'].nunique()
    n_sensors = len(SENSOR_COLUMNS)
    n_positive = int(df['target'].sum())
    n_negative = int(len(df) - n_positive)
    pos_pct = (n_positive / len(df)) * 100
    neg_pct = (n_negative / len(df)) * 100
    imbalance_ratio = n_negative / n_positive if n_positive > 0 else 0

    engine_lifespans = df.groupby('unit_number')['time_in_cycles'].max()
    min_life = int(engine_lifespans.min())
    max_life = int(engine_lifespans.max())
    mean_life = float(engine_lifespans.mean())
    median_life = float(engine_lifespans.median())

    feature_cols = [
        col for col in df.columns 
        if col not in INDEX_COLUMNS + ['RUL', 'target']
    ]

    summary_content = f"""================================================================================
NASA C-MAPSS FD001 DATASET & PREPROCESSING SUMMARY
Predictive Maintenance ML Capstone Project - Step 2
================================================================================

1. BASIC DATASET METRICS
--------------------------------------------------------------------------------
- Total Processed Rows       : {n_rows:,}
- Total Processed Columns    : {n_cols}
- Unique Engines (Units)     : {n_engines} (Unit 1 to Unit {n_engines})
- Total Raw Sensors          : {n_sensors} (sensor_1 through sensor_21)
- Active/Informative Sensors : {len(INFORMATIVE_SENSORS)}
- Constant/Inactive Sensors  : {len(CONSTANT_SENSORS)}
- Operational Settings       : {len(SETTING_COLUMNS)} (op_setting_1, op_setting_2, op_setting_3)

2. ENGINE LIFESPAN METRICS (Full Run-to-Failure Trajectories)
--------------------------------------------------------------------------------
- Minimum Lifespan (cycles)  : {min_life} (Engine {engine_lifespans.idxmin()})
- Maximum Lifespan (cycles)  : {max_life} (Engine {engine_lifespans.idxmax()})
- Mean Lifespan (cycles)     : {mean_life:.2f}
- Median Lifespan (cycles)   : {median_life:.2f}
- Total Operational Cycles   : {df['time_in_cycles'].sum():,}

3. TARGET LABEL & FAILURE RISK CONFIGURATION
--------------------------------------------------------------------------------
- Failure Window Threshold   : {FAILURE_WINDOW} cycles (target = 1 if RUL <= {FAILURE_WINDOW} else 0)
- Failure Risk Samples (1)   : {n_positive:,} ({pos_pct:.2f}%)
- Normal Operation Samples (0): {n_negative:,} ({neg_pct:.2f}%)
- Class Imbalance Ratio      : {imbalance_ratio:.2f}:1 (Negative : Positive)

4. FEATURE ENGINEERING SPECIFICATION
--------------------------------------------------------------------------------
- Base Raw Features Retained : {len(SETTING_COLUMNS) + len(SENSOR_COLUMNS)}
- Engineered Time-Series Feat: {len(feature_cols) - len(SETTING_COLUMNS) - len(SENSOR_COLUMNS)}
- Total Model Input Candidate: {len(feature_cols)}
- Lag & Rolling Mechanics    : Strictly causal backward-looking per engine.
                               Prevents cross-engine contamination and future lookahead.

Engineered Feature Names ({len(feature_cols) - len(SETTING_COLUMNS) - len(SENSOR_COLUMNS)} features):
{', '.join([col for col in feature_cols if col not in SETTING_COLUMNS + SENSOR_COLUMNS])}

5. DATA LEAKAGE & INTEGRITY VERIFICATION
--------------------------------------------------------------------------------
- RUL Column Included For Target Formulation Only (Excluded from ML Input Features)
- Unit ID and Time Cycles Preserved for Grouped Splitting and Evaluation
- Missing / Null Values Across Entire Dataset: {df.isnull().sum().sum()}
================================================================================
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary_content)

    print(f"Data summary successfully written to {output_path}")


def generate_plots(df: pd.DataFrame, output_dir: str = 'outputs/plots'):
    """
    Generate the 4 required exploratory visualizations and save them to output_dir:
    1. 01_failure_distribution.png
    2. 02_rul_distribution.png
    3. 03_engine_cycle_distribution.png
    4. 04_sensor_correlation.png
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # Plot 1: 01_failure_distribution.png
    plt.figure(figsize=(8, 5))
    target_counts = df['target'].value_counts().sort_index()
    labels = ['Healthy (RUL > 30)', 'Failure Risk (RUL <= 30)']
    colors = ['#2b5c8f', '#d9534f']
    bars = plt.bar(labels, target_counts, color=colors, width=0.5, edgecolor='black', linewidth=1)
    
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100
        plt.text(bar.get_x() + bar.get_width()/2., height + 300,
                 f'{height:,}\n({pct:.1f}%)',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    plt.title(f'Failure Risk Target Distribution (Window = {FAILURE_WINDOW} Cycles)', fontsize=13, fontweight='bold', pad=15)
    plt.ylabel('Observation Count', fontsize=11)
    plt.ylim(0, len(df) * 0.95)
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, '01_failure_distribution.png')
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved: {plot1_path}")

    # Plot 2: 02_rul_distribution.png
    plt.figure(figsize=(9, 5))
    sns.histplot(df['RUL'], bins=40, kde=True, color='#1f77b4', edgecolor='black', alpha=0.6)
    plt.axvline(x=FAILURE_WINDOW, color='#d9534f', linestyle='--', linewidth=2.5,
                label=f'Failure Threshold (RUL = {FAILURE_WINDOW})')
    plt.title('Distribution of Remaining Useful Life (RUL) Across All Engine Cycles', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Remaining Useful Life (Cycles)', fontsize=11)
    plt.ylabel('Density / Frequency', fontsize=11)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, '02_rul_distribution.png')
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved: {plot2_path}")

    # Plot 3: 03_engine_cycle_distribution.png
    plt.figure(figsize=(9, 5))
    engine_max_cycles = df.groupby('unit_number')['time_in_cycles'].max()
    sns.histplot(engine_max_cycles, bins=25, kde=True, color='#2ca02c', edgecolor='black', alpha=0.6)
    plt.axvline(engine_max_cycles.mean(), color='#ff7f0e', linestyle='-', linewidth=2,
                label=f'Mean Lifespan: {engine_max_cycles.mean():.1f} cycles')
    plt.axvline(engine_max_cycles.median(), color='#9467bd', linestyle=':', linewidth=2,
                label=f'Median Lifespan: {engine_max_cycles.median():.1f} cycles')
    plt.title('Distribution of Maximum Operational Cycles (Lifespan per Engine)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Total Cycles Until Failure', fontsize=11)
    plt.ylabel('Number of Engines', fontsize=11)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    plt.tight_layout()
    plot3_path = os.path.join(output_dir, '03_engine_cycle_distribution.png')
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved: {plot3_path}")

    # Plot 4: 04_sensor_correlation.png
    plt.figure(figsize=(12, 10))
    # Select informative sensors + RUL + target
    corr_cols = INFORMATIVE_SENSORS + ['RUL', 'target']
    corr_matrix = df[corr_cols].corr()
    
    # Custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=cmap, vmin=-1, vmax=1,
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8})
    plt.title('Sensor Correlation Matrix with RUL and Failure Target', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plot4_path = os.path.join(output_dir, '04_sensor_correlation.png')
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"Saved: {plot4_path}")


def run_pipeline(
    raw_data_path: str = 'data/train_FD001.txt',
    processed_output_path: str = 'data/processed_fd001.csv',
    summary_path: str = 'outputs/data_summary.txt',
    plots_dir: str = 'outputs/plots'
) -> pd.DataFrame:
    """
    Execute full Step 2 preprocessing pipeline.
    """
    print(">>> [1/6] Loading raw FD001 dataset...")
    df = load_dataset(raw_data_path)
    print(f"    Loaded raw shape: {df.shape}")

    print(">>> [2/6] Sorting chronologically by unit_number and time_in_cycles...")
    df = sort_chronologically(df)

    print(">>> [3/6] Handling and validating missing values...")
    df = handle_missing_values(df)

    print(">>> [4/6] Calculating Remaining Useful Life (RUL) and Binary Failure Target...")
    df = calculate_rul(df)
    df = create_failure_target(df, failure_window=FAILURE_WINDOW)

    print(">>> [5/6] Generating time-aware lag and rolling features...")
    df = create_engineered_features(df, sensors=INFORMATIVE_SENSORS, rolling_windows=[5])
    print(f"    Processed dataset shape after feature engineering: {df.shape}")

    print(">>> [6/6] Saving processed dataset and artifacts...")
    os.makedirs(os.path.dirname(processed_output_path), exist_ok=True)
    df.to_csv(processed_output_path, index=False)
    print(f"    Saved processed dataset to: {processed_output_path}")

    generate_data_summary(df, output_path=summary_path)
    generate_plots(df, output_dir=plots_dir)

    print(">>> Pipeline completed successfully!")
    return df


if __name__ == '__main__':
    run_pipeline()
