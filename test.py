import pandas as pd
from classification import classification


df = pd.read_csv("/home/reza/Documents/sample.csv")

classification(df, "class")