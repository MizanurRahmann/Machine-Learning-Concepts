from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

iris = datasets.load_iris()
print(type(iris))
print(iris.keys())
print("(Sample, features) === ",iris.data.shape)
print("Target names are: ", iris.target_names)

X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
print(df.head())


