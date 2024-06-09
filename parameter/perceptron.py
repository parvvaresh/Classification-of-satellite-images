param_perceptron = {
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha_l1': [0.0001, 0.001, 0.01, 0.1],  # Alpha for l1 penalty
    'alpha_l2': [0.0001, 0.001, 0.01, 0.1],  # Alpha for l2 penalty
    'learning_rate': ['constant', 'optimal', 'adaptive'],
    'eta0': [0.01, 0.1, 0.5]  # Initial learning rate
}
