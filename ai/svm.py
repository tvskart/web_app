from skimage.morphology import skeletonize
from sklearn import datasets, svm, metrics
from sklearn.externals import joblib
digits = datasets.load_digits()

images_and_labels = list(zip(digits.images, digits.target))

n_samples = len(digits.images)
data = digits.images

data = data.reshape((n_samples, -1))
classifier = svm.SVC(gamma=0.001,degree=4)

classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])
filename = 'svm.sav'
joblib.dump(classifier, filename)
expected = digits.target[n_samples / 2:]
predicted = classifier.predict(data[n_samples / 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
