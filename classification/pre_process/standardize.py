import numpy as np
from sklearn.preprocessing import (
                                    StandardScaler,
                                    MinMaxScaler,
                                    MaxAbsScaler,
                                    RobustScaler,
                                    Normalizer)



def standardize(X : np.array) -> list:
  scaler_standard = StandardScaler()
  X_standard_scaled = scaler_standard.fit_transform(X)

  scaler_minmax = MinMaxScaler()
  X_minmax_scaled = scaler_minmax.fit_transform(X)

  scaler_maxabs = MaxAbsScaler()
  X_maxabs_scaled = scaler_maxabs.fit_transform(X)

  scaler_robust = RobustScaler()
  X_robust_scaled = scaler_robust.fit_transform(X)

  scaler_normalizer = Normalizer()
  X_normalized = scaler_normalizer.fit_transform(X)

  return [X_standard_scaled,
          X_minmax_scaled,
          X_maxabs_scaled,
          X_robust_scaled,
          X_normalized]
