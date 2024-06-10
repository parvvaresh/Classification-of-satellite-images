from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def select_best_model(X_train, y_train, models):

    best_model = None
    best_params = None
    best_score = 0
    metrics = {}
    
    for model_name, (model, param_grid) in models.items():
        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        metrics[model_name] = {}
        
        for params, mean_score, scores in zip(
            grid_search.cv_results_['params'],
            grid_search.cv_results_['mean_test_score'],
            grid_search.cv_results_['std_test_score']
        ):
            model.set_params(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_train)
            
            accuracy = accuracy_score(y_train, y_pred)
            precision = precision_score(y_train, y_pred)
            recall = recall_score(y_train, y_pred)
            f1 = f1_score(y_train, y_pred)
            
            metrics[model_name][str(params)] = {
                'mean_score': mean_score,
                'std_score': scores,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            if mean_score > best_score:
                best_model = model_name
                best_params = params
                best_score = mean_score
    
    return best_model, best_params, best_score, metrics


