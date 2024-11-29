# Agricultural Products Classification Pipeline

## Overview
This pipeline is designed to classify agricultural products using satellite data from **SENTINEL-1** and **SENTINEL-2**. The pipeline includes the following stages:

1. **Data Standardization**: Different standardization techniques are applied to the data to make it suitable for model training.
2. **Dimensionality Reduction**: PCA and LDA are applied to reduce the dimensionality of the feature space, with separate models for each satellite's data.
3. **Model Training and Hyperparameter Optimization**: Various machine learning models are trained, and hyperparameter optimization is performed using grid search.


I've added the information you provided to the README. Here's the updated section that includes the satellite data input:

---

### Satellite Data Input:

The input dataset contains Earth observation data from **SENTINEL-1** and **SENTINEL-2** satellites, obtained via Google Earth Engine. The data includes various bands from both satellites, as well as additional values relevant for classification tasks.

#### Example of input data:

| **Sample** | **0_B1**  | **0_B2**  | **0_B3**  | **0_B4**  | **0_B5**  | **0_B6**  | **0_B7**  | **0_B8**  | **0_B8A** | **0_B9**  | **0_B11** | **0_B12** | **0_VV** |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|----------|
| **Sample 1** | 0.050643478 | 0.071909783 | 0.108879348 | 0.140969565 | 0.156472826 | 0.172709783 | 0.185292391 | 0.180054348 | 0.195056522 | 0.205251087 | 0.195241304 | 0.1603 | -1 |
| **Sample 2** | 0.051273684 | 0.07195 | 0.107911842 | 0.138413158 | 0.156592105 | 0.180571053 | 0.195072368 | 0.189626316 | 0.204071053 | 0.243975 | 0.199786842 | 0.161619737 | -1 |
| **Sample 3** | 0.064336805 | 0.097296528 | 0.140022222 | 0.176558333 | 0.187975 | 0.19215 | 0.199796528 | 0.203748611 | 0.201070833 | 0.235688194 | 0.202470833 | -15.741307 | -1 |
| **Sample 4** | 0.070949999 | 0.100846154 | 0.150261539 | 0.196115385 | 0.214473077 | 0.219430769 | 0.227103846 | 0.226692308 | 0.230776923 | 0.23485 | 0.240280769 | 0.209653846 | -1 |
| **Sample 5** | 0.071380468 | 0.101917188 | 0.151620313 | 0.198378125 | 0.213576563 | 0.215678125 | 0.222285156 | 0.224170313 | 0.224170313 | 0.235323438 | 0.235323438 | 0.208569531 | -1 |
| **Sample 6** | 0.072846154 | 0.100773077 | 0.150984615 | 0.198823077 | 0.213915385 | 0.217265385 | 0.224673077 | 0.226946154 | 0.226946154 | 0.234361538 | 0.237073077 | 0.206880769 | -1 |
| **Sample 7** | 0.067707143 | 0.103935714 | 0.152242857 | 0.200014286 | 0.209557143 | 0.213071429 | 0.221978571 | 0.229471429 | 0.223307143 | 0.232307143 | 0.232307143 | 0.205528571 | -1 |
| **Sample 8** | 0.097139552 | 0.130318657 | 0.162661194 | 0.194323881 | 0.209510448 | 0.212884328 | 0.222468657 | 0.230838806 | 0.230782836 | 0.236003731 | 0.311174627 | 0.283676866 | -1 |
| **Sample 9** | 0.070247222 | 0.097663194 | 0.129397222 | 0.159320833 | 0.171659722 | 0.17494375 | 0.183878472 | 0.192720833 | 0.193045833 | 0.276390278 | 0.256345833 | 0.249488889 | -1 |
| **Sample 10** | 0.060408333 | 0.085986806 | 0.121355556 | 0.154906944 | 0.168461111 | 0.1728375 | 0.182507639 | 0.191263889 | 0.192247222 | 0.282597917 | 0.263926389 | 0.249488889 | -1 |



- The **bands** from **SENTINEL-2** include: `B1`, `B2`, `B3`, `B4`, `B5`, `B6`, `B11`, `B12`, etc.
- The **SENTINEL-1** data includes polarization bands such as `VV` and `VH`, with additional derived features such as `VV_1` and `VH_1`.
- Each row represents a specific point in time for the satellite’s data, with `1_VV` marking the timestamp of the observation.


---


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

