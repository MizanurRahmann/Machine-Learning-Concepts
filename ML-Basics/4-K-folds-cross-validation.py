import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

creditData = pd.read_csv("./Dataset/credit_data.csv")

features = creditData[['income', 'age', 'loan']]
target = creditData.default

model = LogisticRegression()
predicted = cross_val_predict(model, features, target, cv=10)

print(accuracy_score(target, predicted) * 100)