import pandas as pd
from classification import classification

import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignore ConvergenceWarning
warnings.filterwarnings("ignore", category = ConvergenceWarning)


df = pd.read_csv("/home/reza/Desktop/ghazvin.csv")
df = df.drop(["Unnamed: 0"], axis=1)
classification(df, "Name")