import pandas as pd
import numpy as np
from .pre_process.data.parser import get_spectrums

def split_data(df: pd.DataFrame) -> list:
    s1_columns = []
    s2_columns = []

    data = get_spectrums()


    for column in df.columns:
        if  check_s1(column):
            s1_columns.append(column)

        elif  check_s2(column):
            s2_columns.append(column)

    s1 = pd.concat([df[col] for col in s1_columns], axis=1)
    s2 = pd.concat([df[col] for col in s2_columns], axis=1)

    return [
            np.array(s1),
            np.array(s2),
    ]


def check_s1(column):
    data = get_spectrums()
    for s1_column in data["s1"]:
        if s1_column in column.upper():
            return True
    return False



def check_s2(column):
    data = get_spectrums()
    for s2_column in data["s2"]:
        if s2_column in column.upper():
            return True
    return False


# not work for S1 and S2
# def extract_spectrum(column: str) -> str:
#     spectrum = ""

#     try:
#         _, spectrum = column.split("_")
#     except ValueError:
#         return spectrum

#     return spectrum
