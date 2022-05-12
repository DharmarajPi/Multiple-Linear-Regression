
#Multiple Linear Regression

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
data=pd.read_csv('Salesdata.csv')
#data.columns

X = data[['TV', 'Radio', 'Newspaper']]
y = data.Sales

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30, random_state = 1)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

print("RMSE :", np.sqrt(mean_squared_error(y_test, lr_preds)))
print("R^2: ", r2_score(y_test, lr_preds))

#Prediction for TV = 121, Radio = 8.4, Newspaper = 48.7
lr.predict([[121,8.4,48.7]])

		
