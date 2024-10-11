import pyspark.sql.functions as F
from pyspark.ml.feature import PCA, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
import numpy as np

spark = SparkSession.builder.appName("PCA").getOrCreate()

def pca_split(X_s1: DataFrame, X_s2: DataFrame) -> DataFrame:
  best_n_components_s1 = get_best_n_components(X_s1)
  best_n_components_s2 = get_best_n_components(X_s2)

  X_pca_s1 = pca(X_s1, best_n_components_s1)
  X_pca_s2 = pca(X_s2, best_n_components_s2)

  X_combined = X_pca_s1.join(X_pca_s2)

  return X_combined

def get_best_n_components(df: DataFrame) -> int:
  feature_cols = df.columns
  assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
  df_vector = assembler.transform(df)

  pca = PCA(k=len(feature_cols), inputCol="features", outputCol="pca_features")
  pca_model = pca.fit(df_vector)
  explained_variance = pca_model.explainedVariance.toArray()

  cumulative_variance = np.cumsum(explained_variance)
  best_n_components = np.argmax(cumulative_variance >= 0.95) + 1

  return best_n_components

def pca(df: DataFrame, n_components_best: int) -> DataFrame:
  feature_cols = df.columns
  assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
  df_vector = assembler.transform(df)

  pca = PCA(k=n_components_best, inputCol="features", outputCol="pca_features")
  pca_model = pca.fit(df_vector)
  df_pca = pca_model.transform(df_vector)

  for i in range(n_components_best):
    df_pca = df_pca.withColumn(f'PCA_{i+1}', F.col("pca_features")[i])

  return df_pca.select([f'PCA_{i+1}' for i in range(n_components_best)])

