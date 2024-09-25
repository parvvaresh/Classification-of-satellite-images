import pandas as pd
from pre_process import pre_process
from train_models import train_models


def classification(df : pd.DataFrame,
         class_column : str) -> None:
    
    x_data , y = pre_process(df, class_column)

    train_models(x_data, y)






