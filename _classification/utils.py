import pandas as pd
import numpy as np
from .pre_process.data.parser import get_spectrums


def split_data(df: pd.DataFrame) -> list:
    """
    Splits the input DataFrame into two subsets (s1 and s2) based on specific conditions.

    Parameters:
        df: pd.DataFrame
            Input DataFrame containing the columns to be split.

    Returns:
        list:
            A list containing two NumPy arrays:
            - s1: Subset of the data matching the "s1" criteria.
            - s2: Subset of the data matching the "s2" criteria.
    """

    # Lists to store column names for s1 and s2
    s1_columns = []
    s2_columns = []

    # Get the predefined spectral data for s1 and s2
    data = get_spectrums()

    # Iterate through each column in the input DataFrame
    for column in df.columns:
        # Check if the column belongs to s1 based on the spectral data
        if check_s1_and_s2(column, data["s1"]):
            s1_columns.append(column)
        # Check if the column belongs to s2 based on the spectral data
        elif check_s1_and_s2(column, data["s2"]):
            s2_columns.append(column)

    # Concatenate the selected columns for s1 and s2 into separate DataFrames
    s1 = pd.concat([df[col] for col in s1_columns], axis=1)
    s2 = pd.concat([df[col] for col in s2_columns], axis=1)

    # Convert the DataFrames to NumPy arrays and return them as a list
    return [
        np.array(s1),  # Subset for s1
        np.array(s2),  # Subset for s2
    ]


def check_s1_and_s2(column: str, data: list) -> bool:
    """
    Checks if a column name matches any item in the given spectral data list.

    Parameters:
        column: str
            The column name to be checked.

        data: list
            A list of predefined spectral data for matching.

    Returns:
        bool:
            True if the column matches any item in the list (case-insensitive), False otherwise.
    """
    # Convert column to uppercase and check if it matches any item in the data list
    for _column in data:
        if _column in column.upper():
            return True
    return False
