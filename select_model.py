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
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")






def classification_parameter_finder(model,
                                    parameters : dict,
                                    X_train : np.array,
                                    y_train : np.array,
                                    X_test : np.array,
                                    y_test : np.array):
    start = time.time()

    # Create a scorer based on Cohen's kappa
    kappa_scorer = make_scorer(cohen_kappa_score)

    # Set GridSearchCV to optimize for kappa
    grid = GridSearchCV(model, param_grid=parameters, refit=True, cv=5, n_jobs=-1, scoring=kappa_scorer)
    grid.fit(X_train, y_train)

    y_train_pred = grid.predict(X_train)
    y_test_pred = grid.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    kappa = cohen_kappa_score(y_test, y_test_pred)

    conf_matrix = confusion_matrix(y_test, y_test_pred)
    class_labels = np.unique(y_test)

    model_name = str(model).split('(')[0]

    end = time.time()

    results = pd.DataFrame({
        "model": [model_name],
        "best_params": [grid.best_params_],
        "train_accuracy": [train_accuracy],
        "test_accuracy": [test_accuracy],
        "precision": [precision],
        "recall": [recall],
        "f1_score": [f1],
        "kappa": [kappa],
        "confusion_matrix": [conf_matrix.ravel()],
        "runtime": [end - start]
    })

    name = f"{model_name}.csv"
    results.to_csv(name)

    print(f"The best parameters for {model_name} model is: {grid.best_params_}")
    print("--" * 10)
    print(f"Accuracy in the training set is {train_accuracy:0.2%} for {model_name} model.")
    print(f"Accuracy in the testing set is {test_accuracy:0.2%} for {model_name} model.")
    print(f"Precision is {precision:0.2%} for {model_name} model.")
    print(f"Recall is {recall:0.2%} for {model_name} model.")
    print(f"F1 Score is {f1:0.2%} for {model_name} model.")
    print(f"Kappa Score is {kappa:0.2%} for {model_name} model.")
    print("--" * 10)
    print(f"Runtime of the program is: {end - start:0.2f} seconds")

    # Plot the confusion matrix
    #plot_confusion_matrix(conf_matrix, class_labels, model_name)