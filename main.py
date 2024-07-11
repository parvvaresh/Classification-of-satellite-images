import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from classification.parameter_finder import classification_parameter_finder
from classification.models.models import get_details_models
from classification.pre_process.standardize import standardize
from classification.utils import split_data
from classification.pre_process.dimensionality_reduction.lda import lda_split, lda
from classification.pre_process.dimensionality_reduction.pca import pca_split, pca


def main(df : pd.DataFrame,
         class_column : str) -> None:
    

    X , y  = df , df[class_column]
    details_models = get_details_models()

    # pre process 1-> standardize data


    standardize_data = standardize(X)

    # pre process 2 -> apply pca and lda

    data = {}
    for name , _data in standardize_data.items():
        s1 , s2 =  split_data(_data)

        data_pca =  pca(_data, None)
        data_lda = lda(_data, None)
        split_pca = pca_split(s1, s2, y)
        split_lda = lda_split(s1, s2, y)

        temp = {
            "original" : _data,
            "pca" : data_pca,
            "lda" : data_lda,
            "split pca" : split_pca,
            "split lda" : split_lda

        }

        data[name] = temp


    results = []

    for  name_section, data_section in data.items():
        for name_subsection , data_subsection in data_section.items():
            method = f"name_section-name_subsection"
            X_train, X_test, y_train, y_test = train_test_split(data_subsection, y, test_size=0.2, random_state=42)

            for detail_model in details_models:
                model, parameters = detail_model


                _result = classification_parameter_finder(model,
                                                parameters,
                                                X_train,
                                                y_train,
                                                X_test,
                                                y_test,
                                                method)
                results.append(_result)


    results = pd.concat(results, ignore_index=True)

    results.to_csv("result.csv")





