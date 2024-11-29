from sklearn.ensemble import RandomForestClassifier

# ******** Get Random Forest Function ********
def get_rf():
    """
    Initializes a Random Forest classifier and provides a parameter grid for hyperparameter tuning.

    Returns:
        tuple:
            - RandomForestClassifier object: An instance of the Random Forest classifier.
            - dict: A dictionary containing hyperparameter options for tuning.
    """

    # Define the hyperparameter grid for the Random Forest classifier
    param_randomForest = {
        'n_estimators': [50, 100, 200],  # Number of trees in the forest
        'criterion': ['gini', 'entropy'],  # Splitting criteria: Gini Impurity or Entropy
        'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees (None means no limit)
        'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
        'min_samples_leaf': [1, 2, 4]  # Minimum samples required at a leaf node
    }

    # Initialize a RandomForestClassifier object
    random_forest = RandomForestClassifier()

    # Return the classifier and the hyperparameter grid
    return random_forest, param_randomForest
