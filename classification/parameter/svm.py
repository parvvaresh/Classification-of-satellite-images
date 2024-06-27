param_svm = {
    'C': [0.1, 1, 10, 100, 1000],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],  # Degree for 'poly' kernel
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'coef0': [0, 0.1, 0.5, 1],  # Coef0 for 'poly' and 'sigmoid' kernels
}