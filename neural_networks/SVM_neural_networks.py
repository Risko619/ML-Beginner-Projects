''' Where KNN gives us very low accuracy score on high dimensional data that doesn't have a linear correspondence'''
''' Support Vector Machines (SVM) works best'''

import sklearn
from sklearn import svm
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import metrics

cancer = load_breast_cancer()

#print("Features: ", cancer.feature_names)
#print("Labels: ", cancer.target_names)

x = cancer.data
y = cancer.target

#x = np.random.rand(569, 30)
#y = np.random.randint(0, 2, size=569)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

print(x_train, y_train)
classes = ['malignant', 'benign']

''' Implementation'''

clf = svm.SVC(kernel = 'linear')
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print(acc)

''' Summary'''

'''In summary, your code loads the breast cancer dataset, splits it into training and testing sets, 
trains a Support Vector Machine classifier on the training data, makes predictions on the test data, and calculates and prints the accuracy of the model.
 The goal is to assess how well the SVM classifier performs on the breast cancer classification task.'''