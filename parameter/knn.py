param_knn = {
    'n_neighbors': list(range(3, 15)),  # Number of neighbor
    'weights': ['uniform', 'distance'],  # Weight function
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm to compute neighbors
    'metric': ['euclidean', 'manhattan', 'chebyshev']  # Distance metric
}