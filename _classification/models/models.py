# Importing model functions from different scripts
from .decision_tree import get_dt  # Decision Tree Classifier
from .knn import get_knn  # K-Nearest Neighbors Classifier
from .logstic_regression import get_lr  # Logistic Regression Classifier
from .mlp import get_mlp  # Multi-Layer Perceptron Classifier
from .naive_bayes import get_nb  # Naive Bayes Classifier
from .perceptron import get_pr  # Perceptron Classifier
from .random_forest import get_rf  # Random Forest Classifier
from .svm import get_svm  # Support Vector Machine Classifier
from .NearestCentroid import get_nc  # Nearest Centroid Classifier

# ******** Get Details of All Models Function ********
def get_details_models():
    """
    Returns a list of all available classification models and their respective hyperparameter grids.
    
    Each model function is called, which returns the model instance and its hyperparameters. 
    The function returns a list of tuples, where each tuple contains a model and its associated parameter grid.

    Returns:
        list: A list of tuples, each containing a classifier and a dictionary of hyperparameters.
    """
    return [
        get_nc(),  # Nearest Centroid
        get_knn(),  # K-Nearest Neighbors
        get_dt(),  # Decision Tree
        get_lr(),  # Logistic Regression
        get_mlp(),  # Multi-Layer Perceptron
        get_nb(),  # Naive Bayes
        get_pr(),  # Perceptron
        get_rf(),  # Random Forest
        get_svm()   # Support Vector Machine
    ]
