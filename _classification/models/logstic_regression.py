from sklearn.linear_model import LogisticRegression

# ******** Get Logistic Regression Function ********
def get_lr():
    """
    Initializes a Logistic Regression classifier and provides a parameter grid for hyperparameter tuning.

    Returns:
        tuple:
            - LogisticRegression object: An instance of the Logistic Regression classifier.
            - dict: A dictionary containing hyperparameter options for tuning.
    """

    # Define the hyperparameter grid for Logistic Regression
    param_logsticRegression = {
        'penalty': ['l1', 'l2'],  # Regularization techniques: L1 (Lasso) or L2 (Ridge)
        'C': [0.01, 0.1, 1, 10, 100],  # Inverse regularization strength (smaller values = stronger regularization)
        'solver': ['liblinear', 'saga'],  # Optimization solvers for fitting the model
        'max_iter': [100, 200, 300, 500]  # Maximum number of iterations for solver convergence
    }

    # Initialize a LogisticRegression object
    logistic_regression = LogisticRegression()

    # Return the classifier and the hyperparameter grid
    return logistic_regression, param_logsticRegression
