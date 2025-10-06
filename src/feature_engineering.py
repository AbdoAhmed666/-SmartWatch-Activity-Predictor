# src/feature_engineering.py
import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame, rolling_window: int = 3) -> pd.DataFrame:
    """
    Add derived features:
    - heart_rate_diff: one-step difference of heart rate
    - hr_roll_mean, hr_roll_std: rolling statistics of heart_rate
    - steps_per_distance: steps / (distance + eps)
    - steps_x_distance (already exists sometimes) kept if present
    - zscore_hr: (hr - resting_heart) / sd_norm_heart if those cols exist
    """
    df2 = df.copy()

    if 'hear_rate' in df2.columns:
        df2['heart_rate_diff'] = df2['hear_rate'].diff().fillna(0)
        # rolling ops (use min periods = 1)
        df2['hr_roll_mean'] = df2['hear_rate'].rolling(window=rolling_window, min_periods=1).mean()
        df2['hr_roll_std'] = df2['hear_rate'].rolling(window=rolling_window, min_periods=1).std().fillna(0)
    else:
        df2['heart_rate_diff'] = 0.0
        df2['hr_roll_mean'] = 0.0
        df2['hr_roll_std'] = 0.0

    if 'distance' in df2.columns and 'steps' in df2.columns:
        eps = 1e-6
        df2['steps_per_distance'] = df2['steps'] / (df2['distance'] + eps)
    else:
        df2['steps_per_distance'] = 0.0

    # if resting and sd present, compute zscore relative to resting heart
    if all(col in df2.columns for col in ['resting_heart', 'sd_norm_heart', 'hear_rate']):
        # avoid divide by zero
        df2['zscore_hr'] = (df2['hear_rate'] - df2['resting_heart']) / (df2['sd_norm_heart'] + 1e-6)
    else:
        df2['zscore_hr'] = 0.0

    # Keep original interaction if present
    if 'steps_times_distance' not in df2.columns and 'steps' in df2.columns and 'distance' in df2.columns:
        df2['steps_times_distance'] = df2['steps'] * df2['distance']

    # Fill any inf/nan
    df2 = df2.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df2
