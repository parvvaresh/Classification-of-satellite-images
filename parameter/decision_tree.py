"""
 explain :
        --> criterion :
                pass
        --> max_depth :
                pass
        --> min_samples_split :
                pass
        --> min_samples_leaf :
                pass
"""

param_decisionTree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


