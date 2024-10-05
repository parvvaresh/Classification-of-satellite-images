import pandas as pd
from sklearn.preprocessing import (
                                    StandardScaler,
                                    MinMaxScaler,
                                    MaxAbsScaler,
                                    RobustScaler,
                                    Normalizer)



def standardize(df : pd.DataFrame) -> list:


  columns = df.columns


  scaler_standard = StandardScaler()
  X_standard_scaled = scaler_standard.fit_transform(df)
  df_standard_scaled = pd.DataFrame(X_standard_scaled, columns=columns)


  scaler_minmax = MinMaxScaler()
  X_minmax_scaled = scaler_minmax.fit_transform(df)
  df_minmax_scaled = pd.DataFrame(X_minmax_scaled, columns=columns)


  scaler_maxabs = MaxAbsScaler()
  X_maxabs_scaled = scaler_maxabs.fit_transform(df)
  df_maxabs_scaled = pd.DataFrame(X_maxabs_scaled, columns=columns)



  scaler_robust = RobustScaler()
  X_robust_scaled = scaler_robust.fit_transform(df)
  df_robust_scaled = pd.DataFrame(X_robust_scaled, columns=columns)


  scaler_normalizer = Normalizer()
  X_normalized = scaler_normalizer.fit_transform(df)
  df_normalized = pd.DataFrame(X_normalized, columns=columns)

  return {
        "orginal" : df,
        "standard_scaled" : df_standard_scaled,
        "minmax_scaled" : df_minmax_scaled,
        "maxabs_scaled" : df_maxabs_scaled,
        "robust_scaled" : df_robust_scaled,
        "normalized" : df_normalized
      }
