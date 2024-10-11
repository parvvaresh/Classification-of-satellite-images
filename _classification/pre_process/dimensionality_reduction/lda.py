import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LinearDiscriminantAnalysis
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("LDA Example") \
    .getOrCreate()

def lda_split(X_s1: pd.DataFrame, 
              X_s2: pd.DataFrame, 
              y: pd.Series) -> pd.DataFrame:
    spark_X_s1 = spark.createDataFrame(X_s1)
    spark_X_s2 = spark.createDataFrame(X_s2)
    spark_y = spark.createDataFrame(y.rename("label"))

    spark_X_s1 = spark_X_s1.withColumn("label", spark_y["label"])
    spark_X_s2 = spark_X_s2.withColumn("label", spark_y["label"])

    best_n_components_s1 = get_best_n_components(spark_X_s1)
    best_n_components_s2 = get_best_n_components(spark_X_s2)

    X_lda_s1 = lda(spark_X_s1, best_n_components_s1)
    X_lda_s2 = lda(spark_X_s2, best_n_components_s2)

    X_combined = X_lda_s1.join(X_lda_s2, on=None)  # Assuming the same indices
    return X_combined

def get_best_n_components(X: pd.DataFrame) -> int:
    # Convert pandas DataFrame to PySpark DataFrame
    spark_X = spark.createDataFrame(X)

    # Vectorize the features
    vector_assembler = VectorAssembler(inputCols=spark_X.columns[:-1], outputCol="features")
    spark_X_vectorized = vector_assembler.transform(spark_X)

    lda = LinearDiscriminantAnalysis()
    lda_model = lda.fit(spark_X_vectorized)
    
    # The maximum number of components for LDA is limited to min(n_classes - 1, n_features)
    n_classes = len(spark_X.select("label").distinct().collect())
    n_features = len(spark_X.columns) - 1
    n_components = min(n_classes - 1, n_features)

    return n_components

def lda(X: pd.DataFrame, n_components_best: int) -> pd.DataFrame:
    # Convert pandas DataFrame to PySpark DataFrame
    spark_X = spark.createDataFrame(X)

    # Vectorize the features
    vector_assembler = VectorAssembler(inputCols=spark_X.columns[:-1], outputCol="features")
    spark_X_vectorized = vector_assembler.transform(spark_X)

    lda = LinearDiscriminantAnalysis(featuresCol="features", labelCol="label", predictionCol="prediction")
    lda_model = lda.fit(spark_X_vectorized)

    # Transform the data using LDA
    X_lda = lda_model.transform(spark_X_vectorized)

    # Select the LDA components and prediction
    X_lda_df = X_lda.select("prediction", "features").toPandas()

    # Convert LDA result to DataFrame with appropriate column labels
    column_labels = [f'LDA_{i + 1}' for i in range(n_components_best)]
    X_lda_df.columns = column_labels

    return X_lda_df

# Example usage
# Assuming X_s1 and X_s2 are pandas DataFrames and y is a pandas Series
X_s1 = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
X_s2 = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
y = pd.Series(np.random.randint(0, 2, size=100))  # Binary target variable

X_combined_lda = lda_split(X_s1, X_s2, y)

print(X_combined_lda.head())
