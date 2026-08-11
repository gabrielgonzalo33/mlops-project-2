"""
clean_data.py

Preprocessing utilities for the churn prediction model.
Handles data loading, missing values, and categorical encoding.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load customer data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the customer data
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records from {filepath}")
    return df


def handle_missing_values(
    df: pd.DataFrame, 
    strategy: str = 'median'
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        strategy: Strategy for handling missing values
                  Options: 'median', 'mean', 'drop'
                  
    Returns:
        DataFrame with missing values handled
    """
    df_clean = df.copy()
    
    # Get numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if strategy == 'median':
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"Filled {col} missing values with median: {median_val:.2f}")
                
    elif strategy == 'mean':
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                mean_val = df_clean[col].mean()
                df_clean[col].fillna(mean_val, inplace=True)
                print(f"Filled {col} missing values with mean: {mean_val:.2f}")
                
    elif strategy == 'drop':
        initial_rows = len(df_clean)
        df_clean.dropna(inplace=True)
        dropped = initial_rows - len(df_clean)
        print(f"Dropped {dropped} rows with missing values")
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'median', 'mean', or 'drop'.")
    
    # Handle categorical missing values (fill with mode)
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            mode_val = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_val, inplace=True)
            print(f"Filled {col} missing values with mode: {mode_val}")
    
    return df_clean


def encode_categorical(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    One-hot encode categorical columns.
    
    Args:
        df: Input DataFrame
        columns: List of columns to encode. If None, encodes all object columns.
        
    Returns:
        DataFrame with categorical columns encoded
    """
    df_encoded = df.copy()
    
    if columns is None:
        columns = df_encoded.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns:
        if col in df_encoded.columns:
            # Create dummy variables
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded.drop(col, axis=1, inplace=True)
            print(f"Encoded {col} into {len(dummies.columns)} dummy variables")
    
    return df_encoded


def scale_features(
    df: pd.DataFrame, 
    columns: List[str],
    method: str = 'standard'
) -> pd.DataFrame:
    """
    Scale numeric features.
    
    Args:
        df: Input DataFrame
        columns: List of columns to scale
        method: Scaling method ('standard' or 'minmax')
        
    Returns:
        DataFrame with scaled features
    """
    df_scaled = df.copy()
    
    for col in columns:
        if col in df_scaled.columns:
            if method == 'standard':
                mean = df_scaled[col].mean()
                std = df_scaled[col].std()
                df_scaled[col] = (df_scaled[col] - mean) / std
            elif method == 'minmax':
                min_val = df_scaled[col].min()
                max_val = df_scaled[col].max()
                df_scaled[col] = (df_scaled[col] - min_val) / (max_val - min_val)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            print(f"Scaled {col} using {method} scaling")
    
    return df_scaled


if __name__ == "__main__":
    # Example usage
    print("Preprocessing module loaded successfully.")
    print("Available functions: load_data, handle_missing_values, encode_categorical, scale_features")
