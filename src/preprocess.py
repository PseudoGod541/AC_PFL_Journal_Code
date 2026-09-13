#preprocess.py
import numpy as np
import pandas as pd
from typing import Tuple, List

def calculate_rul(df: pd.DataFrame, rul_cap: int = 125) -> pd.DataFrame:
    """
    Calculates the Remaining Useful Life (RUL) for each engine.
    Uses a piecewise linear degradation model and includes input validation.

    Args:
        df (pd.DataFrame): The input dataframe.
        rul_cap (int): The maximum RUL value to cap at. Defaults to 125.

    Returns:
        pd.DataFrame: DataFrame with an added 'RUL' column.
    """
    # --- Input validation ---
    if df.empty:
        raise ValueError("Input DataFrame for calculate_rul is empty.")
    required_cols = ['unit_id', 'cycle']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame is missing required columns: {required_cols}")
    
    max_cycles = df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    
    df = df.merge(max_cycles, on='unit_id', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df = df.drop(columns=['max_cycle'])
    
    # Cap RUL at the specified value
    df['RUL'] = df['RUL'].clip(upper=rul_cap)
    return df

def create_sequences(df: pd.DataFrame, feature_cols: List[str], sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms the data into sequences for time-series prediction.
    Includes input validation and handles units with insufficient data.
    
    Returns:
        A tuple of (X, y) numpy arrays.
    """
    # --- Input validation ---
    if df.empty:
        raise ValueError("Input DataFrame for create_sequences is empty.")
    if not all(col in df.columns for col in feature_cols + ['RUL']):
        raise ValueError("Input DataFrame is missing required feature or RUL columns.")

    X, y = [], []
    skipped_units = 0
    
    for unit_id in df['unit_id'].unique():
        unit_data = df[df['unit_id'] == unit_id]
        
        # Skip units with fewer cycles than the sequence length
        if len(unit_data) < sequence_length:
            skipped_units += 1
            continue
            
        for i in range(len(unit_data) - sequence_length + 1):
            X.append(unit_data[feature_cols].iloc[i:i+sequence_length].values)
            y.append(float(unit_data['RUL'].iloc[i+sequence_length-1]))
            
    if skipped_units > 0:
        print(f"⚠️  Skipped {skipped_units} units with insufficient data for sequence creation.")
    
    # --- Validation for empty results ---
    if not X:
        raise ValueError("No sequences could be created. Check data and sequence_length parameter.")

    X_arr, y_arr = np.array(X), np.array(y)
    
    print(f"✅ Created sequences: X shape = {X_arr.shape}, y shape = {y_arr.shape}")
            
    return X_arr, y_arr


def load_test_data(test_file: str, rul_file: str, feature_cols: list, 
                   scaler, sequence_length: int = 30):
    """
    Loads test set and ground truth RUL for final evaluation.
    Takes LAST sequence per engine (standard C-MAPSS protocol).
    """
    df_test = load_cmapss_data(test_file)
    rul_true = pd.read_csv(rul_file, header=None, names=['RUL'])
    
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])
    
    X_test = []
    for unit_id in sorted(df_test['unit_id'].unique()):
        unit_data = df_test[df_test['unit_id'] == unit_id][feature_cols].values
        if len(unit_data) >= sequence_length:
            X_test.append(unit_data[-sequence_length:])  # last window only
        else:
            # pad with zeros if insufficient
            pad = np.zeros((sequence_length - len(unit_data), len(feature_cols)))
            X_test.append(np.vstack([pad, unit_data]))
    
    return np.array(X_test), rul_true['RUL'].values
    
def nasa_rul_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Asymmetric NASA scoring function.
    Late predictions (positive error) penalized exponentially harder.
    """
    error = y_pred - y_true
    scores = np.where(error < 0, np.exp(-error / 13) - 1, np.exp(error / 10) - 1)
    return float(np.sum(scores))