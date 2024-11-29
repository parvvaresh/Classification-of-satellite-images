from sklearn.neighbors import KNeighborsClassifier

# ******** Get K-Nearest Neighbors Function ********
def get_knn():
    """
    Initializes a K-Nearest Neighbors (KNN) classifier and provides a parameter grid for hyperparameter tuning.

    Returns:
        tuple: 
            - KNeighborsClassifier object: An instance of the K-Nearest Neighbors Classifier.
            - dict: A dictionary containing hyperparameter options for tuning.
    """

    # Define the hyperparameter grid for the KNN classifier
    param_knn = {
        'n_neighbors': list(range(1, 15, 2)),  # Number of neighbors to consider, ranging from 1 to 15 with a step of 2
        'weights': ['uniform', 'distance'],  # Weighting scheme: 'uniform' (all points equal) or 'distance' (inverse distance)
        'metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metrics to use for the KNN algorithm
    }

    # Initialize a KNeighborsClassifier object
    knn = KNeighborsClassifier()

    # Return the classifier and the hyperparameter grid
    return knn, param_knn
