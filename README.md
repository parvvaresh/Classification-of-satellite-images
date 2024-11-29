Here’s a refined version of the README file to ensure clarity and proper structure:

---

# Agricultural Products Classification Pipeline

## Overview
This pipeline is designed to classify agricultural products using satellite data from **SENTINEL-1** and **SENTINEL-2**. The pipeline includes the following stages:

1. **Data Standardization**: Different standardization techniques are applied to the data to make it suitable for model training.
2. **Dimensionality Reduction**: PCA and LDA are applied to reduce the dimensionality of the feature space, with separate models for each satellite's data.
3. **Model Training and Hyperparameter Optimization**: Various machine learning models are trained, and hyperparameter optimization is performed using grid search.

## Pipeline Steps

### 1. Data Standardization
The following standardization methods are applied to the data:

- **Original**: Raw data without scaling.
- **Standard Scaled**: Standardization using mean and variance.
- **MinMax Scaled**: Scales data to a specified range (usually [0,1]).
- **MaxAbs Scaled**: Scales data to [-1, 1] based on the maximum absolute value.
- **Robust Scaled**: Scales data using the median and interquartile range.
- **Normalized**: Scales data to unit norm.

### 2. Dimensionality Reduction
Two dimensionality reduction techniques are used:

- **PCA (Principal Component Analysis)**: Reduces the feature space by projecting the data into a lower-dimensional space.
- **LDA (Linear Discriminant Analysis)**: A classification-specific dimensionality reduction technique.

Note: Each satellite’s data is processed separately due to different band spaces. For **SENTINEL-1**, the bands are:
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

Each model is optimized based on its hyperparameter grid.

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

Here’s the updated **Setup** section with the commands:

---

## Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/parvvaresh/Classification-of-satellite-images.git
    cd your-repository-name
    ```

2. **Install dependencies:**

    First, create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

    Then, install the required Python libraries:

    ```bash
    pip install -r requirements.txt
    ```





## Usage

The pipeline is designed to handle satellite data, perform preprocessing, apply dimensionality reduction techniques, and train various models with optimized hyperparameters.

To use the pipeline, follow the steps below:

1. Prepare your input data as a CSV file with the necessary features and target column.
2. Modify the input data and parameters in the respective scripts to suit your specific agricultural classification problem.

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
classification(df, "ClassColumn", "/home/reza/data_test", "data_test")
```

### Explanation:
- `pre_process(df, class_column)` processes the data, separating the features (`x_data`) and target variable (`y`).
- `train_models(x_data, y, path, name)` trains machine learning models and saves the trained models to the specified path (`path`) using the provided name (`name`).

### Output:
After training, the results will be saved into a CSV file containing the following information:
- **Method**: The data standardization and dimensionality reduction method used.
- **Model name**: The name of the model.
- **Best hyperparameters**: The best hyperparameters found during grid search.
- **Train accuracy**: Accuracy on the training dataset.
- **Test accuracy**: Accuracy on the test dataset.
- **Precision**, **Recall**, **F1 Score**, **Kappa**: Metrics for model evaluation.
- **Confusion Matrix path**: Path to the confusion matrix plot.
- **Runtime**: The time taken to train the model.
- **Best model**: The best model with its parameters.

---

### Sample Result Entry:

| method              | model                  | best_params                                                                                       | train_accuracy | test_accuracy | precision | recall | f1_score | kappa | confusion_matrix_path | runtime | best_model                                   |
|---------------------|------------------------|--------------------------------------------------------------------------------------------------|----------------|----------------|-----------|--------|----------|-------|------------------------|---------|----------------------------------------------|
| original-original   | KNeighborsClassifier    | `{'metric': 'euclidean', 'n_neighbors': 1, 'weights': 'uniform'}`                                | 1.0            | 0.925          | 0.909     | 0.925  | 0.913    | 0.903 | path_to_matrix.png      | 2.43    | KNeighborsClassifier(metric='euclidean', n_neighbors=1) |

---

