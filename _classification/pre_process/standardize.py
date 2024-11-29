import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    Normalizer
)

# ******** Standardize Function ********
def standardize(df: pd.DataFrame) -> dict:
    """
    Applies various data scaling techniques to the input DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame to be scaled.
        
    Returns:
        dict: A dictionary containing the original and scaled DataFrames using different scalers.
    """
    
    # Get the column names of the input DataFrame
    columns = df.columns

    # ******** Standard Scaling ********
    # StandardScaler scales features by removing the mean and scaling to unit variance.
    scaler_standard = StandardScaler()  # Initialize the StandardScaler
    X_standard_scaled = scaler_standard.fit_transform(df)  # Fit and transform the data
    df_standard_scaled = pd.DataFrame(X_standard_scaled, columns=columns)  # Create a DataFrame

    # ******** Min-Max Scaling ********
    # MinMaxScaler scales features to a range between 0 and 1.
    scaler_minmax = MinMaxScaler()  # Initialize the MinMaxScaler
    X_minmax_scaled = scaler_minmax.fit_transform(df)  # Fit and transform the data
    df_minmax_scaled = pd.DataFrame(X_minmax_scaled, columns=columns)  # Create a DataFrame

    # ******** Max-Abs Scaling ********
    # MaxAbsScaler scales each feature by its maximum absolute value, preserving sparsity.
    scaler_maxabs = MaxAbsScaler()  # Initialize the MaxAbsScaler
    X_maxabs_scaled = scaler_maxabs.fit_transform(df)  # Fit and transform the data
    df_maxabs_scaled = pd.DataFrame(X_maxabs_scaled, columns=columns)  # Create a DataFrame

    # ******** Robust Scaling ********
    # RobustScaler scales features using statistics that are robust to outliers 
    # (e.g., median and interquartile range).
    scaler_robust = RobustScaler()  # Initialize the RobustScaler
    X_robust_scaled = scaler_robust.fit_transform(df)  # Fit and transform the data
    df_robust_scaled = pd.DataFrame(X_robust_scaled, columns=columns)  # Create a DataFrame

    # ******** Normalization ********
    # Normalizer scales each sample (row) to have unit norm, preserving the shape of the data.
    scaler_normalizer = Normalizer()  # Initialize the Normalizer
    X_normalized = scaler_normalizer.fit_transform(df)  # Fit and transform the data
    df_normalized = pd.DataFrame(X_normalized, columns=columns)  # Create a DataFrame

    # Return all scaled DataFrames along with the original DataFrame
    return {
        "original": df,  # Original DataFrame
        "standard_scaled": df_standard_scaled,  # Standard-scaled DataFrame
        "minmax_scaled": df_minmax_scaled,  # MinMax-scaled DataFrame
        "maxabs_scaled": df_maxabs_scaled,  # MaxAbs-scaled DataFrame
        "robust_scaled": df_robust_scaled,  # Robust-scaled DataFrame
        "normalized": df_normalized  # Normalized DataFrame
    }
