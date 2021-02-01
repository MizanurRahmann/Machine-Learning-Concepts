import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

# Read .csv file into a Dataframe
dataset = pd.read_csv('./Dataset/house_prices.csv')
size = dataset['sqft_living']
price = dataset['price']

# Converts dataframe to array
x = np.array(size).reshape(-1, 1)
y = np.array(price).reshape(-1, 1)

# Use Linear Regression + fit() => H(x) = b0 + b1*x
model = LinearRegression()
model.fit(x, y)

# MSE and R value
regression_model_mse = mean_squared_error(x, y)
print("MSE: ", math.sqrt(regression_model_mse))
print("R squared value: ", model.score(x, y))

# Get the coef (b0)
print(model.coef_[0])
# Get intercept (b1)
print(model.intercept_[0])

# Visulaize the dataset with the fitted model
plt.scatter(x, y, color="green")
plt.plot(x, model.predict(x), color="red")
plt.title("Linear Regression")
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()

# Predictiong the price
print("Predicted by model: ", model.predict([[2000]]))
