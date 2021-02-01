import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# π = 1 / 1 + exp[ - ( b0 + b1 * x )]

x1 = np.array([0,0.6,1.1,1.5,1.8,2.5,3,3.1,3.9,4,4.9,5,5.1])
y1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])

x2 = np.array([3,3.8,4.4,5.2,5.5,6.5,6,6.1,6.9,7,7.9,8,8.1])
y2 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])

X = np.array([[0],[0.6],[1.1],[1.5],[1.8],[2.5],[3],[3.1],[3.9],[4],[4.9],[5],[5.1],[3],[3.8],[4.4],[5.2],[5.5],[6.5],[6],[6.1],[6.9],[7],[7.9],[8],[8.1]])
y = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1])

plt.plot(x1, y1, 'ro', color='blue')
plt.plot(x2, y2, 'ro', color='red')

model = LogisticRegression()
model.fit(X, y)

b0 = model.intercept_
b1 = model.coef_

print("b0 is: ", b0)
print("b1 is: ", b1)

def sigmoid(classifier, x):
    return 1/(1 + np.exp(-(b0 + b1 * x)))

for i in range(1, 120):
    plt.plot(i/10.0 - 2, sigmoid(model, i/10.0), 'ro', color="green")

plt.axis([-2, 10, -0.5, 2])
plt.show()

pred = model.predict_proba([[6]])
print("Prediction: ", pred*100)