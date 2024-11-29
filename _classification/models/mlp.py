from sklearn.neural_network import MLPClassifier

# ******** Get Multi-Layer Perceptron (MLP) Classifier Function ********
def get_mlp():
    """
    Initializes a Multi-Layer Perceptron (MLP) classifier and provides a parameter grid for hyperparameter tuning.

    Returns:
        tuple:
            - MLPClassifier object: An instance of the Multi-Layer Perceptron classifier.
            - dict: A dictionary containing hyperparameter options for tuning.
    """

    # Define the hyperparameter grid for the MLP classifier
    param_mlp = {
        'hidden_layer_sizes': [
            (50,), (100,), (50, 50), (100, 100)
        ],  # Number of neurons in each hidden layer, e.g., single-layer 50 or two-layers 50-50
        'activation': ['tanh', 'relu'],  # Activation functions: 'tanh' or 'ReLU'
        'solver': ['sgd', 'adam'],  # Optimization solvers: Stochastic Gradient Descent (SGD) or Adam
        'alpha': [0.0001, 0.001, 0.01],  # L2 regularization parameter to prevent overfitting
        'learning_rate': ['constant', 'adaptive'],  # Learning rate schedule
        'max_iter': [100, 200, 300, 400, 500],  # Maximum number of iterations to converge
    }

    # Initialize an MLPClassifier object
    mlp_classifier = MLPClassifier()

    # Return the classifier and the hyperparameter grid
    return mlp_classifier, param_mlp
