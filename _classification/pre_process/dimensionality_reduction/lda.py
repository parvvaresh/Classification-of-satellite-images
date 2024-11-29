from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

# ******** LDA Split Function ********
def lda_split(X_s1: np.array, X_s2: np.array, y: np.array) -> np.array:
    """
    Perform Linear Discriminant Analysis (LDA) on two datasets separately 
    and combine their transformed components.

    Parameters:
        X_s1 (np.array): The first dataset (2D array) for LDA.
        X_s2 (np.array): The second dataset (2D array) for LDA.
        y (np.array): The target labels (1D array) associated with the datasets.

    Returns:
        np.array: A combined array of LDA-transformed features from both datasets.
    """

    # Get the best number of components for the first dataset
    best_n_components_s1 = get_best_n_components(X_s1, y)
    # Get the best number of components for the second dataset
    best_n_components_s2 = get_best_n_components(X_s2, y)

    # Apply LDA to the first dataset with the optimal number of components
    X_lda_s1 = lda(X_s1, y, best_n_components_s1)
    # Apply LDA to the second dataset with the optimal number of components
    X_lda_s2 = lda(X_s2, y, best_n_components_s2)

    # Combine the LDA-transformed datasets by horizontally stacking them
    X = np.hstack((X_lda_s1, X_lda_s2))
    return X

# ******** Get Best Number of Components ********
def get_best_n_components(X: np.array, y: np.array) -> int:
    """
    Determine the optimal number of LDA components that explain at least 95% of the variance.

    Parameters:
        X (np.array): The dataset (2D array) to analyze.
        y (np.array): The target labels (1D array) associated with the dataset.

    Returns:
        int: The optimal number of components.
    """

    # Initialize the LDA object
    lda = LinearDiscriminantAnalysis()

    # Fit LDA to the data and target labels
    X_lda = lda.fit_transform(X, y)

    # Compute the explained variance ratio
    explained_variance_ratio = lda.explained_variance_ratio_

    # Compute the cumulative variance ratio
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Determine the smallest number of components that explain at least 95% of the variance
    n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1

    return n_components

# ******** LDA Function ********
def lda(X: np.array, y: np.array, n_components_best: int) -> np.array:
    """
    Apply Linear Discriminant Analysis (LDA) on the dataset with a specified number of components.

    Parameters:
        X (np.array): The dataset (2D array) to transform.
        y (np.array): The target labels (1D array) associated with the dataset.
        n_components_best (int): The number of components to retain.

    Returns:
        np.array: The LDA-transformed dataset.
    """

    # If the number of components is not provided, compute the optimal number of components
    if isinstance(n_components_best, type(None)):
        n_components_best = get_best_n_components(X, y)

    # Initialize the LDA object with the specified number of components
    lda = LinearDiscriminantAnalysis(n_components=n_components_best)

    # Fit LDA to the data and transform it
    X_lda = lda.fit_transform(X, y)
    return X_lda
