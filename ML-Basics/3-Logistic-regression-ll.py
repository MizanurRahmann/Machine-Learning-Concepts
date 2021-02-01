import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Read data from data table
creditData = pd.read_csv("./Dataset/credit_data.csv")

print(creditData.head()) # view first 5 data from data table
print(creditData.describe()) # describe basic information
print(creditData.corr()) # corelation with each other

# Set features and target field
features = creditData[["income", "age", "loan"]]
target = creditData.default

# Split data to train and test dataset
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3)

# Apply Logistic regression for classification
model = LogisticRegression()
model.fit(feature_train, target_train)
predictions = model.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions) * 100)