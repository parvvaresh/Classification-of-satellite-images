from sklearn.svm import SVC

# ******** Get Support Vector Machine (SVM) Function ********
def get_svm():
    """
    Initializes a Support Vector Classifier (SVC) and provides a parameter grid for hyperparameter tuning.

    Returns:
        tuple:
            - SVC object: An instance of the Support Vector Classifier.
            - dict: A dictionary containing hyperparameter options for tuning.
    """

    # Define the hyperparameter grid for the SVC (Support Vector Classifier)
    param_svm = {
        'C': [0.1, 1, 10, 100, 1000],  # Regularization parameter (larger values mean less regularization)
        'kernel': ['rbf'],  # The kernel type to be used in the algorithm (radial basis function kernel)
        'gamma': [0.001, 0.01, 0.1, 1],  # Kernel coefficient for 'rbf' kernel
    }

    # Initialize a Support Vector Classifier object
    svm_classifier = SVC()

    # Return the classifier and the hyperparameter grid
    return svm_classifier, param_svm
