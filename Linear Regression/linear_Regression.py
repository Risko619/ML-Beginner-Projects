''' This version is how we Save Models, how to plot data on a grid, and visualize some of what we're doing '''
''' We save our high percentage models so we can use on future model sets'''
''' Make sure you "pip install matplotlib.plt" in the terminal'''

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

# import matplot lib, pickle, and change style of our grid
import matplotlib.pyplot as plt
import pickle
import os
from matplotlib import style

df = pd.read_csv('/Volumes/Leroy/Software Engineering/Machine Learning projects/Linear Regression/student-mat.csv')
data = pd.read_csv('/Volumes/Leroy/Software Engineering/Machine Learning projects/Linear Regression/student-mat.csv', sep=";")
print(data.head())
 
data = data[['G1', 'G2', 'G3', 'absences']]

predict = "G3"

X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

''' Changes made from here to 40, is so we can find a better percentage model than previous'''
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
best = 0
for var in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

if acc > best:
    best = acc
# Here's how we save the model
    with open('studentmodel.pickle', 'wb') as f:
        pickle.dump(linear, f)

# Read in our pickle file in 'rb' mode
pickle_in = open('studentmodel.pickle', 'rb')
# Now we load this pickle in our linear model
linear = pickle.load(pickle_in)
                     
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# Scatter plotting our data

p = 'G1'
style.use('ggplot')
plt.scatter(data[p], data['G3'])
plt.xlabel(p)
plt.ylabel('Final Grade')
plt.show()

''' To change the results shown on pyplot, we would change the "p =" variable'''
