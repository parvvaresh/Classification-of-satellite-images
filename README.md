# Agricultural Products Classification Pipeline

## Overview
This pipeline is designed to classify agricultural products using satellite data from two satellites, **SENTINEL-1** and **SENTINEL-2**. The pipeline consists of several stages, including:

1. **Data Standardization**: Various standardization techniques are applied to the data.
2. **Dimensionality Reduction**: PCA and LDA are used for reducing the dimensionality of the feature space, with separate models applied for each satellite's data.
3. **Model Training and Hyperparameter Optimization**: Different machine learning models are trained and their hyperparameters are tuned using grid search.

## Pipeline Steps

### 1. Data Standardization
The following standardization methods are applied to the data:

- **Original**: Raw data without scaling.
- **Standard Scaled**: Standardization using mean and variance.
- **MinMax Scaled**: Scaling data to a specific range (usually [0,1]).
- **MaxAbs Scaled**: Scales data to [-1, 1] based on the maximum absolute value.
- **Robust Scaled**: Scales using the median and interquartile range.
- **Normalized**: Scales data to unit norm.

### 2. Dimensionality Reduction
Two dimensionality reduction techniques are used:

- **PCA (Principal Component Analysis)**: Applied to reduce the feature space.
- **LDA (Linear Discriminant Analysis)**: Used for classification-specific dimensionality reduction.

Note: Each satellite's data is processed separately as their band spaces differ. For **SENTINEL-1**, the bands are:
- `VH`, `VV`, `HH`, `VH_1`, `VV_1`

For **SENTINEL-2**, the bands are:
- `B1`, `B2`, `B3`, `B4`, `B5`, `B6`, `B11`, `B12`, `B13`, `B14`, `B15`, `B16`, `NDVI`, `EVI`, `SAVI`

### 3. Model Training & Hyperparameter Optimization
The following models are trained using grid search for hyperparameter optimization:

1. **Decision Tree Classifier**
2. **K-Nearest Neighbors (KNN)**
3. **Logistic Regression**
4. **Multilayer Perceptron (MLP)**
5. **Naive Bayes**
6. **Nearest Centroid**
7. **Perceptron**
8. **Random Forest**
9. **Support Vector Machine (SVM)**

Each model has its respective hyperparameter grid to optimize.

---

## Model Hyperparameters

| Model                  | Hyperparameters                                                                                                                                                       |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Decision Tree**       | `criterion`: ['gini', 'entropy'], `max_depth`: [None, 10, 20, 30, 40], `min_samples_split`: [2, 5, 10], `min_samples_leaf`: [1, 2, 4]                                  |
| **K-Nearest Neighbors** | `n_neighbors`: [1, 3, 5, 7, 9, 11, 13], `weights`: ['uniform', 'distance'], `metric`: ['euclidean', 'manhattan', 'minkowski']                                      |
| **Logistic Regression** | `penalty`: ['l1', 'l2'], `C`: [0.01, 0.1, 1, 10, 100], `solver`: ['liblinear', 'saga'], `max_iter`: [100, 200, 300, 500]                                           |
| **MLP**                 | `hidden_layer_sizes`: [(50,), (100,), (50, 50), (100, 100)], `activation`: ['tanh', 'relu'], `solver`: ['sgd', 'adam'], `alpha`: [0.0001, 0.001, 0.01], `max_iter`: [100, 200, 300, 400, 500] |
| **Naive Bayes**         | `var_smoothing`: [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]                                                                                                                     |
| **Nearest Centroid**    | `metric`: ['euclidean', 'manhattan'], `shrink_threshold`: [None, 0.1, 0.2, 0.5, 0.7, 0.8]                                                                      |
| **Perceptron**          | `penalty`: ['l1', 'l2', 'elasticnet'], `alpha`: [0.0001, 0.001, 0.01, 0.1, 1], `max_iter`: [1000, 2000, 3000]                                                      |
| **Random Forest**       | `n_estimators`: [50, 100, 200], `criterion`: ['gini', 'entropy'], `max_depth`: [None, 10, 20, 30], `min_samples_split`: [2, 5, 10], `min_samples_leaf`: [1, 2, 4]   |
| **SVM**                 | `C`: [0.1, 1, 10, 100, 1000], `kernel`: ['rbf'], `gamma`: [0.001, 0.01, 0.1, 1]                                                                                   |

---

## Requirements

- Python 3.x
- pandas
- scikit-learn
- numpy
- matplotlib (for plotting and visualization)

## Setup

1. Clone the repository.
2. Install the dependencies using `pip install -r requirements.txt`.
3. Run the scripts in the following order:
   - Data preprocessing (standardization and dimensionality reduction).
   - Model training and hyperparameter tuning.

## Usage

- The pipeline is designed to handle satellite data, perform preprocessing, apply dimensionality reduction techniques, and train various models with optimized hyperparameters.
- Modify the input data or parameters in the respective scripts to fit your specific agricultural classification problem.

---

## Example Usage

```python
import pandas as pd
from pre_process import pre_process
from train_models import train_models

def classification(df: pd.DataFrame,
                   class_column: str,
                   path: str,
                   name: str) -> None:
    """
    This function takes a DataFrame, preprocesses the data, 
    and trains models on the processed data.
    
    Parameters:
    - df: pandas DataFrame containing the data to be classified.
    - class_column: The column name containing the target variable.
    - path: The path where the trained model and results will be saved.
    - name: The name used to save the model and results.
    """
    # Preprocess the data to separate features and target variable
    x_data, y = pre_process(df, class_column)

    # Train models using the processed data
    train_models(x_data, y, path, name)

# Example usage
path_csv = "/data.csv"  # Path to your dataset
df = pd.read_csv(path_csv)  # Read the CSV into a DataFrame

# Call the classification function
classification(df, "Name", "/home/reza/data_test", "data_test")
```

### Explanation:
- `pre_process(df, class_column)` processes the data and splits it into features (`x_data`) and target variable (`y`).
- `train_models(x_data, y, path, name)` trains machine learning models on the processed data and saves the trained models to the specified path (`path`) using the provided name (`name`).
  
Modify the `path_csv`, `class_column`, and output paths to match your specific setup. This code serves as a template for running the pipeline with your own dataset.