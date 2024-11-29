from sklearn.naive_bayes import GaussianNB

# ******** Get Naive Bayes Function ********
def get_nb():
    """
    Initializes a Gaussian Naive Bayes classifier and provides a parameter grid for hyperparameter tuning.

    Returns:
        tuple:
            - GaussianNB object: An instance of the Gaussian Naive Bayes classifier.
            - dict: A dictionary containing hyperparameter options for tuning.
    """

    # Define the hyperparameter grid for Gaussian Naive Bayes
    param_naive_bayes = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        # 'var_smoothing': A smoothing parameter added to the variance to prevent zero probabilities
        # and handle numerical stability issues. Values closer to 1e-9 are typical defaults.
    }

    # Initialize a GaussianNB object
    naive_bayes = GaussianNB()

    # Return the classifier and the hyperparameter grid
    return naive_bayes, param_naive_bayes
