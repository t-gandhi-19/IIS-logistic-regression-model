
import sklearn
import pandas as pd
import numpy as np

from numpy import mean
from numpy import std
from pandas import read_csv

from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv("DataMl.csv", sep=",")

# separate into input and output columns
X = df.drop(['ID','DOB','12graduation','CollegeID','CollegeCityID','High-Salary'],axis=1).values   # independant features
y = df['High-Salary'].values
l = []

for i in range(3998):
    a = max(X[i][22], X[i][16], X[i][17], X[i][18], X[i][19], X[i][20], X[i][21])
    l.append([a])
arr = np.array(l)
X = np.concatenate((X,arr),axis=1)

t = [('a', OneHotEncoder(), [0, 4, 6, 7, 10]), ('b', MinMaxScaler(feature_range = (0,1)), [1,3,5,9,8,12,13,14,15,28]),('c', MinMaxScaler(feature_range = (-1,1)),[23,24,25,26,27]), ('d', MinMaxScaler(feature_range = (0,10)),[11])]
transformer = ColumnTransformer(transformers=t)
#also dropped 10board 
X = transformer.fit_transform(X)

# split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3998)

# transform training data
#X_train = transformer.fit_transform(X_train)
#X_test = transformer.fit_transform(X_test)

label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)
# define the model
model = LogisticRegression(max_iter=1000)
# fit on the training set
model.fit(X_train, y_train)
# predict on test set
yhat = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.2f' % (accuracy*100))

cm = confusion_matrix(y_test, yhat)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm.diagonal())

target_names = ['class 0', 'class 1']
print(classification_report(y_test, yhat, target_names=target_names, digits=4))
