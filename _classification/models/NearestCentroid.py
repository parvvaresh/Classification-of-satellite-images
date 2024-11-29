from sklearn.neighbors import NearestCentroid

# ******** Get Nearest Centroid Function ********
def get_nc():
    """
    Initializes a Nearest Centroid classifier and provides a parameter grid for hyperparameter tuning.

    Returns:
        tuple:
            - NearestCentroid object: An instance of the Nearest Centroid classifier.
            - dict: A dictionary containing hyperparameter options for tuning.
    """

    # Define the hyperparameter grid for the Nearest Centroid classifier
    param_NearestCentroid = {
        'metric': ['euclidean', 'manhattan'],  # Distance metrics to compute nearest centroid
        'shrink_threshold': [None, 0.1, 0.2, 0.5, 0.7, 0.8]
        # 'shrink_threshold': Optional shrinkage threshold to regularize centroids (if not None).
        # Helps to improve robustness with high-dimensional data.
    }

    # Initialize a NearestCentroid object
    nearest_centroid = NearestCentroid()

    # Return the classifier and the hyperparameter grid
    return nearest_centroid, param_NearestCentroid
