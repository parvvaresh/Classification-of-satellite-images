import pandas as pd

from classification.pre_process.standardize import standardize
from classification.utils import split_data
from classification.pre_process.dimensionality_reduction.lda import lda_split, lda
from classification.pre_process.dimensionality_reduction.pca import pca_split, pca



def pre_process(df : pd.DataFrame,
                class_column : str) -> dict:
    


    X , y  = df.drop(class_column, axis=1) , df[class_column]


    print("📌 Start pre process ...")



    print("--- 📌start standardize")
    standardize_data = standardize(X)
    print("--- ✅finish standardize")

    # pre process 2 -> apply pca and lda



    print("--- 📌start dimensionality reduction")

    data = {}
    for name , _data in standardize_data.items():
        s1 , s2 =  split_data(_data)

        data_pca =  pca(_data, None)
        data_lda = lda(_data, y,  None)
        split_pca = pca_split(s1, s2)
        split_lda = lda_split(s1, s2, y)

        temp = {
            "original" : _data,
            "pca" : data_pca,
            "lda" : data_lda,
            "split pca" : split_pca,
            "split lda" : split_lda

        }

        data[name] = temp

    print("--- ✅finish dimensionality reduction")

    print("✅finish pre process ...")


    return data