import pandas as pd
from sklearn.model_selection import train_test_split

from _classification.parameter_finder import classification_parameter_finder
from _classification.models.models import get_details_models

import warnings
from sklearn.exceptions import ConvergenceWarning
import os

# Ignore ConvergenceWarning
warnings.filterwarnings("ignore", category = ConvergenceWarning)


def train_models(x_data : dict,
                 y : pd.DataFrame,
                 path : str,
                 name : str):

    """
    This function trains multiple machine learning models on various subsets of the input dataset, 
    performs hyperparameter tuning, evaluates the models, and saves the results.

    Parameters:
        x_data: dict
            A dictionary containing the feature data split into sections and subsections 
            (e.g., {"section1": {"subsection1": X_data, "subsection2": X_data}}).
            Each subsection represents a different feature subset for training the models.
        
        y: pd.DataFrame
            The target labels for the dataset.
        
        path: str
            The directory path where output files, such as confusion matrix images, will be saved.
        
        name: str
            A descriptive name for the experiment (currently unused but can be used for logging or saving results).

    Workflow:
        - Splits each subsection of data into training and testing sets (80/20 split).
        - Iterates over a list of models and their respective hyperparameters.
        - Calls `classification_parameter_finder` to train, tune, and evaluate each model.
        - Stores the results for each model and dataset combination.

    Output:
        - Results are saved in a CSV file named `result.csv` in the local directory.
        - Confusion matrices and related artifacts are saved to the specified `path`.
        - Prints progress and completion messages to the console for monitoring.

    Notes:
        - This function assumes that the `get_details_models()` function provides a list of tuples, 
          each containing a model instance and its corresponding hyperparameter grid.
        - Suppresses warnings related to model convergence (e.g., ConvergenceWarning).
    """
    

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

        
    # Retrieve the list of models and their hyperparameter configurations
    details_models = get_details_models()

    # Initialize an empty list to store the results of all models
    results = []

    print("📌start train model ...")

    # Iterate through each section of the feature data in x_data
    for name_section, data_section in x_data.items():
        # Iterate through each subsection of the section
        for name_subsection, data_subsection in data_section.items():
            # Define the method name based on the section and subsection
            method = f"{name_section} - {name_subsection}"
            
            # Split the current subsection data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(data_subsection, y, test_size=0.2, random_state=42)

            # Iterate through each model and its corresponding parameters
            for detail_model in details_models:
                model, parameters = detail_model

                print(f"--- 📌start train <<{model}>> on <<{method}>> data")

                # Train and evaluate the model
                _result = classification_parameter_finder(model,
                                                          parameters,
                                                          X_train,
                                                          y_train,
                                                          X_test,
                                                          y_test,
                                                          method,
                                                          path)
                
                print(f"--- ✅finish train <<{model}>> on <<{method}>> data")
                results.append(_result)

    print("✅finish train model")

    # Combine all results into a single DataFrame
    results = pd.concat(results, ignore_index=True)

    # Define the full path for saving the result file
    result_file_path = os.path.join(path, f"{name}.csv")
    
    # Save the results DataFrame to the specified path
    results.to_csv(result_file_path, index=False)

    print(f"             ✅save result in {result_file_path}✅              ")
