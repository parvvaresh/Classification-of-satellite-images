import pandas as pd
from _classification.parameter_finder import classification_parameter_finder
from _classification.models.models import get_details_models

import warnings
from sklearn.exceptions import ConvergenceWarning
import os

# Suppress ConvergenceWarnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def train_models(x_data: dict,
                 y: pd.DataFrame,
                 path: str,
                 name: str,
                 X_train: pd.DataFrame, 
                 X_test: pd.DataFrame, 
                 y_train: pd.DataFrame, 
                 y_test: pd.DataFrame):
    """
    Trains multiple machine learning models on subsets of data, tunes parameters, and saves results.

    Parameters:
        x_data (dict): A dictionary containing feature data, organized into sections and subsections.
                       Example: {"section1": {"subsection1": X_data, "subsection2": X_data}}.
        y (pd.DataFrame): Target labels for the dataset.
        path (str): Directory to save results, confusion matrices, and other artifacts.
        name (str): Name of the experiment, used for naming result files.
        X_train, X_test (pd.DataFrame): Training and testing feature sets.
        y_train, y_test (pd.DataFrame): Training and testing target sets.

    Outputs:
        - Saves results to a CSV file named <name>.csv in the specified path.
        - Saves confusion matrices and related files in the path.
        - Prints progress updates and completion messages.

    Workflow:
        1. Iterates through sections and subsections of `x_data`.
        2. Retrieves model details and hyperparameters from `get_details_models`.
        3. Trains each model using `classification_parameter_finder`.
        4. Combines results into a single DataFrame and saves them.

    Notes:
        - Assumes `classification_parameter_finder` handles model training, tuning, and evaluation.
        - Creates the output directory if it doesn't exist.
        - Suppresses warnings for convergence issues.
    """
    # Ensure the output directory exists
    os.makedirs(path, exist_ok=True)
    print(f"Using directory for outputs: {path}")

    # Retrieve model details (list of tuples with model instances and hyperparameter grids)
    details_models = get_details_models()

    # Initialize a list to store results
    results = []

    print("📌 Starting model training...")

    # Iterate through sections and subsections in the input data
    for name_section, data_section in x_data.items():
        for name_subsection, data_subsection in data_section.items():
            # Define the method name for this data subset
            method = f"{name_section} - {name_subsection}"

            # Iterate through each model and parameter grid
            for detail_model in details_models:
                model, parameters = detail_model

                print(f"--- 📌 Training {model} on {method} data...")

                # Train and evaluate the model
                result = classification_parameter_finder(model=model,
                                                         parameters=parameters,
                                                         X_train=X_train,
                                                         y_train=y_train,
                                                         X_test=X_test,
                                                         y_test=y_test,
                                                         method=method,
                                                         path=path)
                
                print(f"--- ✅ Completed training {model} on {method} data.")
                results.append(result)

    print("✅ Model training completed.")

    # Combine all results into a single DataFrame
    results_df = pd.concat(results, ignore_index=True)

    # Save results to a CSV file
    result_file_path = os.path.join(path, f"{name}.csv")
    results_df.to_csv(result_file_path, index=False)

    print(f"✅ Results saved to {result_file_path}")
