from sklearn.tree import DecisionTreeClassifier

# ******** Get Decision Tree Function ********
def get_dt():
    """
    Initializes a Decision Tree Classifier and provides a parameter grid for hyperparameter tuning.

    Returns:
        tuple: 
            - DecisionTreeClassifier object: An instance of the Decision Tree Classifier.
            - dict: A dictionary containing hyperparameter options for tuning.
    """

    # Define the hyperparameter grid for the Decision Tree Classifier
    param_decisionTree = {
        'criterion': ['gini', 'entropy'],  # Criterion for splitting ('gini' impurity or 'entropy' for information gain)
        'max_depth': [None, 10, 20, 30, 40],  # Maximum depth of the tree (None means unlimited depth)
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
    }

    # Initialize a DecisionTreeClassifier object
    decision_tree = DecisionTreeClassifier()

    # Return the classifier and the hyperparameter grid
    return decision_tree, param_decisionTree
