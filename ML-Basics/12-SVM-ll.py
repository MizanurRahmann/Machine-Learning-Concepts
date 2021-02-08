from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets

features = datasets.load_iris().data
targets = datasets.load_iris().target

featureTrain, featureTest, targetTrain, targetTest = train_test_split(features, targets, test_size=0.3)

#model = svm.SVC(gamma=0.001, C=100)
model = svm.SVC()
fittedModel = model.fit(featureTrain, targetTrain)
predictions = fittedModel.predict(featureTest)

print(confusion_matrix(targetTest, predictions))
print(accuracy_score(targetTest, predictions))

