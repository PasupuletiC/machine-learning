# machine-learning
Simple Linear Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("/content/Salary_Data (1).csv")
dataset.head()
dataset.isna().sum()
x = dataset.iloc[:, :1].values
y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0 )
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
y_pred
plt.scatter(x_train,y_train, color= "red")
plt.plot(x_train, regressor.predict(x_train), color= "blue")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
