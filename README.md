## Summary of the Google Colab Notebook

The notebook contains Python code for various machine learning tasks and models. Here is an overview of its content:

### 1. Libraries and Tools
The notebook begins with the import of the following libraries and tools:
- pandas
- numpy
- sklearn (with multiple sub-modules)
- seaborn
- matplotlib.pyplot

### 2. Data Loading and Exploration
The notebook performs data loading and exploration tasks, including:
- Reading a CSV file named "creditcard.csv" located in "/content/drive/MyDrive/Ai/data/" into a DataFrame called 'df'.
- Displaying the first 5 rows of the DataFrame.
- Providing the number of rows and columns in the dataset using the shape attribute.
- Showing information about the DataFrame using the info() method, which includes data types, non-null values, and memory usage.
- Generating descriptive statistics of the DataFrame using the describe() method, which provides measures of central tendency, dispersion, and distribution shape.

### 3. Label Distribution Analysis
Next, the notebook examines the distribution of labels in the dataset. It performs the following steps:
- Creates a DataFrame called 'labels_count' that counts the occurrences of each label (Class column).
- Displays the table of label counts.
- Plots a countplot to visualize the label distribution.

### 4. Data Preparation
The notebook carries out data preparation steps, including:
- Defining a function called 'find_range' to find the range of features in the DataFrame.
- Applying the 'find_range' function to the DataFrame to display the range of each column.
- Balancing the labels in the dataset by keeping an equal number of samples for each class.
- Creating a new DataFrame called 'df' with the balanced data.
- Displaying the label counts and plotting a countplot for the balanced dataset.
- Splitting the data into training and testing sets (80% train, 20% test) using train_test_split from sklearn.model_selection.
- Normalizing the feature values using MinMaxScaler from sklearn.preprocessing.
- Applying the scaler to the training and testing sets separately.
- Using the 'find_range' function again to display the range of features after normalization.

### 5. Machine Learning Models
The notebook builds and evaluates three machine learning models:
- Logistic Regression:
  - Creating a LogisticRegression model.
  - Fitting the model to the training data.
  - Making predictions on the testing data and calculating the accuracy.
  - Displaying the accuracy score.
  - Showing the confusion matrix using the conf_matrix function.
  - Printing the classification report, including precision, recall, f1-score, and support.

- Support Vector Machines (SVM):
  - Creating an SVC model.
  - Fitting the model to the training data.
  - Making predictions on the testing data and calculating the accuracy.
  - Displaying the accuracy score.
  - Showing the confusion matrix using the conf_matrix function.
  - Printing the classification report, including precision, recall, f1-score, and support.

- K-means:
  - Creating a KMeans model with 2 clusters.
  - Fitting the model to the entire dataset (x).
  - Assigning cluster labels to the data points.
  - Calculating the accuracy by comparing the predicted labels with the original labels (y).
  - Displaying the accuracy score.
  - Showing the confusion matrix using the conf_matrix function.
  - Printing the classification report, including precision, recall, f1-score, and support.

### 6. Model Comparison and Visualization
Finally, the notebook compares the accuracy of the three models and visualizes the results using a bar plot.
