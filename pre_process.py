import pandas as pd

from _classification.pre_process.standardize import standardize
from _classification.utils import split_data
from _classification.pre_process.dimensionality_reduction.lda import lda_split, lda
from _classification.pre_process.dimensionality_reduction.pca import pca_split, pca

def pre_process(df: pd.DataFrame,
                class_column: str) -> list:
    """
    This function preprocesses the input data by standardizing features, 
    performing dimensionality reduction using PCA and LDA, and splitting the data.
    
    Parameters:
        df: pd.DataFrame
            The input DataFrame containing features and the class/target column.
        
        class_column: str
            The name of the column representing the target variable.
    
    Returns:
        x_data: dict
            A dictionary containing the original, PCA, LDA, and split versions of the data.
        
        y: pd.Series
            The target variable extracted from the input DataFrame.
    """
    
    # Separate features (X) and target variable (y)
    X, y = df.drop(class_column, axis=1), df[class_column]

    print("📌 Start pre process ...")

    # Step 1: Standardize the feature data
    print("--- 📌start standardize")
    standardize_data = standardize(X)  # Standardize the input features
    print("--- ✅finish standardize")

    # Step 2: Perform dimensionality reduction (PCA and LDA) and split the data
    print("--- 📌start dimensionality reduction")

    # Initialize an empty dictionary to store processed data
    x_data = {}

    # Iterate through each standardized dataset (if multiple types of standardization are applied)
    for name, _data in standardize_data.items():
        # Split the standardized data into two parts (e.g., for training and testing)
        s1, s2 = split_data(_data)

        # Apply PCA to the entire dataset
        data_pca = pca(_data, None)
        # Apply LDA to the entire dataset
        data_lda = lda(_data, y, None)

        # Apply PCA to the split datasets
        split_pca = pca_split(s1, s2)
        # Apply LDA to the split datasets
        split_lda = lda_split(s1, s2, y)

        # Create a temporary dictionary to store the original, PCA, LDA, and split data
        temp = {
            "original": _data,    # Original standardized data
            "pca": data_pca,      # PCA-reduced data
            "lda": data_lda,      # LDA-reduced data
            "split pca": split_pca,  # PCA-reduced split data (PCA on S1 | pn S2)
            "split lda": split_lda   # LDA-reduced split data (LDA on S1 | pn S2)
        }

        # Add the processed data for the current type of standardization to the main dictionary
        x_data[name] = temp

    print("--- ✅finish dimensionality reduction")

    print("✅finish pre process ...")

    # Return the processed data (x_data) and target variable (y)
    return x_data, y
