from sklearn.linear_model import Perceptron

# ******** Get Perceptron Function ********
def get_pr():
    """
    Initializes a Perceptron classifier and provides a parameter grid for hyperparameter tuning.

    Returns:
        tuple:
            - Perceptron object: An instance of the Perceptron classifier.
            - dict: A dictionary containing hyperparameter options for tuning.
    """

    # Define the hyperparameter grid for the Perceptron classifier
    param_perceptron = {
        'penalty': ['l1', 'l2', 'elasticnet'],  # Regularization types to prevent overfitting:
        # - 'l1': Lasso regularization (sparsity of features).
        # - 'l2': Ridge regularization (shrinks coefficients to reduce multicollinearity).
        # - 'elasticnet': Combination of L1 and L2 regularization.
        
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1],  # Regularization strength (smaller values = stronger regularization).
        
        'max_iter': [1000, 2000, 3000]  # Maximum number of passes over the training data.
    }

    # Initialize a Perceptron classifier
    perceptron = Perceptron()

    # Return the classifier and the hyperparameter grid
    return perceptron, param_perceptron
