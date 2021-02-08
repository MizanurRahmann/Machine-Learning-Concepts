import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# The digits dataset
digits = datasets.load_digits()
images = digits.images
targets = digits.target

# Reshape the features
images = images.reshape( len(digits.images), -1 )


'''
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:6]):
    plt.subplot(2, 3, index+1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Target: %i' % label)
'''

# Split data into train and test datasets
featureTrain, featureTest, targetTrain, targetTest = train_test_split(images, targets, test_size=0.75)

# Train using Support Vector Machine (SVM)
model = svm.SVC()
fittedModel = model.fit(featureTrain, targetTrain)

# Predict the result and measure Confusion Matrix
predictions = fittedModel.predict(featureTest)
print("Confusion matrix:\n%s" % metrics.confusion_matrix(targetTest, predictions))
print(accuracy_score(targetTest, predictions))


# let's test on the last few images
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
print("Prediction for test image: ", model.predict(images[-1].reshape(1,-1)))

plt.show()