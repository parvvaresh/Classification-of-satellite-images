from pre_process.utils import split_data
from pre_process.standardize import standardize
from pre_process.dimensionality_reduction.lda import lda_split, lda
from pre_process.dimensionality_reduction.lda import pca_split, pca

import pandas as pd
import numpy as np






def _split_satellite(data : pd.DataFrame) -> dict:
  s1 , s2 = split_data(data)
  return {
      "orinal" : data,
      "s1" : s1,
      "s2" : s2
    }


def _standardize_data(data  : dict) -> dict:

  result = {}

  for name, _data in data.items():
    X_standard_scaled, X_minmax_scaled, X_maxabs_scaled, X_robust_scaled, X_normalized = standardize(_data)
    result[name] = {
        "orginal" : _data,
        "standard_scaled" : X_standard_scaled,
        "minmax_scaled" : X_minmax_scaled,
        "maxabs_scaled" : X_maxabs_scaled,
        "robust_scaled" : X_robust_scaled,
        "normalized" : X_normalized
      }
  return result



def _dimensionality_reduction(data : dict,
                              y : dict) -> dict:
  result = {}

  for  name, _data_dict in data.items():
    result_temp_pca= {}
    result_temp_lda= {}

    if name == "orginal":
      for _name , _data in _data_dict.items():
        _result_pca = pca(_data, None)
        result_temp_pca[_name] = _result_pca

        _result_lda = lda(_data, y, None)
        result_temp_lda[_name] = _result_lda

      result[f"{name}-PCA"] =  result_temp_pca
      result[f"{name}-LDA"] = result_temp_lda

    else:

      for _name , _data in _data_dict.items():
        _result_pca = pca_split(_data)
        result_temp_pca[_name] = _result_pca

        _result_lda = lda_split(_data, y)
        result_temp_lda[_name] = _result_lda

      result[f"{name}-PCA"] =  result_temp_pca
      result[f"{name}-LDA"] = result_temp_lda



