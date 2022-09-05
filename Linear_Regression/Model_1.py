from sklearn.datasets import load_boston 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names) 
boston['MEDV'] = boston_dataset.target
x = boston[['RM']]
y = boston['MEDV']
model = LinearRegression()
X_train , X_test, Y_train , Y_test = train_test_split(x, y, test_size = 0.7)
model.fit(X_train, Y_train)
y_test_predict = model.predict(X_test)
print(f"mean squared error (MSE) :  {mean_squared_error(Y_test, y_test_predict)}")