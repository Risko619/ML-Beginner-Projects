''' Classification models can't be used on non-numerical data that consists of "yes" or "no" results'''

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv('car.data')
print(data.head())

# This is going to take the labels in our data, and encode to appropriate integer values
le = preprocessing.LabelEncoder()
# Convert our numpy array (list) into integar values for it to work
buying = le.fit_transform(data['buying'])
maint = le.fit_transform(data['maint'])
door = le.fit_transform(data['door'])
persons = le.fit_transform(data['persons'])
lug_boot = le.fit_transform(data['lug_boot'])
safety = le.fit_transform(data['safety'])
cls = le.fit_transform(data['class'])

# print(buying) to see integars of "buying" at bottom of Terminal

# optional - predict = 'class'

# Now we need to recombine our data into a feature list and a label list. We can use the zip() function to makes things easier.
X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

''' IMPLEMENTATION'''

# Now we train and play with our model to get a high accuracy read, by playing with number in "n_neighbors="
model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

# Lets see what the data points are, what's our prediction, and what the actual value is.

predicted = model.predict(x_test)
names = ['unacc','acc', 'good', 'vgood']

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)

