import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    recall_score,
    accuracy_score,
    precision_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
    make_scorer
)
import numpy as np
import pandas as pd
import time
import os  # For handling file paths
import matplotlib.pyplot as plt
import seaborn as sns  # For a more aesthetic plot

# Suppress all warnings
warnings.filterwarnings("ignore")


def classification_parameter_finder(model,
                                    parameters: dict,
                                    X_train: np.array,
                                    y_train: np.array,
                                    X_test: np.array,
                                    y_test: np.array,
                                    method: str,
                                    path: str):


    """
    This function performs hyperparameter tuning for a given classification model using GridSearchCV,
    evaluates its performance on training and testing datasets, and visualizes the confusion matrix.

    Parameters:
        model: sklearn estimator
            The machine learning model to be tuned (e.g., RandomForestClassifier, SVC, etc.).
        
        parameters: dict
            The dictionary containing hyperparameters and their possible values for GridSearchCV.
        
        X_train: np.array
            Training feature data.
        
        y_train: np.array
            Training labels.
        
        X_test: np.array
            Testing feature data.
        
        y_test: np.array
            Testing labels.
        
        method: str
            The name of the method or experiment (used for labeling and saving files).
        
        path: str
            Directory path where output files, such as the confusion matrix image, will be saved.

    Returns:
        results: pandas.DataFrame
            A DataFrame summarizing the best model, its hyperparameters, evaluation metrics
            (accuracy, precision, recall, F1-score, kappa score), and runtime information.
            It also includes the file path of the saved confusion matrix image.
    """



    
    model_name = str(model).split('(')[0]

    start = time.time()


    kappa_scorer = make_scorer(cohen_kappa_score)

    grid = GridSearchCV(model,
                        param_grid=parameters,
                        refit=True,
                        cv=5,
                        n_jobs=-1,
                        scoring=kappa_scorer)
    grid.fit(X_train, y_train)

    y_train_pred = grid.predict(X_train)
    y_test_pred = grid.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    kappa = cohen_kappa_score(y_test, y_test_pred)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred, normalize='true')
    class_labels = np.unique(y_test)

    # Save confusion matrix as an image
    conf_matrix_path = os.path.join(path, f"{model_name}_{method}_confusion_matrix.png")
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"Confusion Matrix - {method}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(conf_matrix_path)
    plt.close()  # Close the plot to avoid overwriting in subsequent calls


    end = time.time()

    # Store results in a DataFrame
    results = pd.DataFrame({
        "method": [method],
        "model": [model_name],
        "best_params": [grid.best_params_],
        "train_accuracy": [train_accuracy],
        "test_accuracy": [test_accuracy],
        "precision": [precision],
        "recall": [recall],
        "f1_score": [f1],
        "kappa": [kappa],
        "confusion_matrix_path": [conf_matrix_path],
        "runtime": [end - start],
        "best_model": [grid.best_estimator_]
    })

    return results
