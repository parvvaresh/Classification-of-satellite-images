import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ******** PCA Split Function ********
def pca_split(X_s1: np.array, X_s2: np.array) -> np.array:
    """
    Perform PCA on two datasets separately and combine their transformed components.

    Parameters:
        X_s1 (np.array): The first dataset (2D array) to apply PCA.
        X_s2 (np.array): The second dataset (2D array) to apply PCA.

    Returns:
        np.array: A combined array of PCA-transformed features from both datasets.
    """

    # Get the best number of components for the first dataset
    best_n_components_s1 = get_best_n_components(X_s1)
    # Get the best number of components for the second dataset
    best_n_components_s2 = get_best_n_components(X_s2)

    # Apply PCA to the first dataset with the optimal number of components
    X_pca_s1 = pca(X_s1, best_n_components_s1)
    # Apply PCA to the second dataset with the optimal number of components
    X_pca_s2 = pca(X_s2, best_n_components_s2)

    # Combine the PCA-transformed datasets by horizontally stacking them
    X = np.hstack((X_pca_s1, X_pca_s2))
    return X

# ******** Get Best Number of Components ********
def get_best_n_components(X: np.array) -> int:
    """
    Determine the optimal number of PCA components that explain at least 95% of the variance.

    Parameters:
        X (np.array): The dataset (2D array) to analyze.

    Returns:
        int: The optimal number of components.
    """

    # Fit PCA on the dataset without specifying the number of components
    pca = PCA().fit(X)

    # Compute the cumulative variance explained by each component
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find the smallest number of components that explain at least 95% of the variance
    best_n_components = np.argmax(cumulative_variance >= 0.95) + 1
    return best_n_components

# ******** PCA Function ********
def pca(X: np.array, n_components_best: int) -> np.array:
    """
    Apply PCA on the dataset with a specified number of components.

    Parameters:
        X (np.array): The dataset (2D array) to transform.
        n_components_best (int): The number of components to retain.
    
    Returns:
        np.array: The PCA-transformed dataset.
    """

    # Check if the number of components is not provided; compute the best number of components
    if isinstance(n_components_best, type(None)):
        n_components_best = get_best_n_components(X)

    # Initialize the PCA object with the specified number of components
    pca = PCA(n_components=n_components_best, svd_solver='auto')

    # Fit PCA to the dataset and transform it
    X_pca = pca.fit_transform(X)
    return X_pca
