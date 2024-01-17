#Activate and install packages
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

#Read the data
df = pd.read_csv('/Volumes/Leroy/Software Engineering/Machine Learning projects/Linear Regression/student-mat.csv')
data = pd.read_csv('/Volumes/Leroy/Software Engineering/Machine Learning projects/Linear Regression/student-mat.csv', sep=";")
print(data.head())
 
data = data[['G1', 'G2', 'G3', 'absences']]

#Define what label we're trying to predict
predict = "G3"

# This is our train data. It Return's us a new df which doesn't have 'G3' in it
X = np.array(data.drop([predict], axis=1))
# This is the attributes
y = np.array(data[predict])

# Here we split 10% of our data into test samples to test
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# In this section we're going to create and see how well our training model is working. Code in our 'Best-Fit' line (Linear Regression)
linear = linear_model.LinearRegression()
# This fits our data to find our best-fit line
linear.fit(x_train, y_train)
# Here we check how accurate our model is
acc = linear.score(x_test, y_test)
print(acc)
# People often stop here once they've received their accuracy.

#-- Now we use and test it on data to see what we get

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])