"""
First I did feature selection, so I removed ‘ID’, ‘DOB’, ‘12graduation’, 'CollegeID', ‘CollegeCityID’.
I decided to remove them as ‘ID’ and ‘DOB’ are not relevant to calculate if a person will get a high salary or not.
‘12graduation’ is like ‘GraduationYear’ and also it doen’t affect our model much(the accuracy of the model is almost the same).  'CollegeID' and ‘CollegeCityID’ is sort of captured by ‘CollegeCityTier’ and ‘CollegeState’.
Also removed ‘10board’ as at 75-25 split accuracy improved from 72.10 to 72.40 after removing.
Did not remove ‘CollegeState’ as accuracy dropped to 72.40 from 72.10 after removing.

1)For columns ‘ComputerProgramming’, ‘ElectronicsAndSemicon’, ‘ComputerScience’, ‘MechanicalEngg’, ‘ElectricalEngg’, ‘TelecomEngg’, ‘CivilEngg’
  I only took the max of the IDs marks in all these subjects and normalised it to values between 0 and 1.

2) one-hot encoded columns ‘Gender’, ‘12board’, ‘Degree’, ‘Specialization’, ‘CollegeState’,

3) Normalised columns ‘10percentage’, ‘12percentage’, ‘CollegeTier’, ‘collegeGPA’, ‘CollegeCityTier’, ‘English’, ‘Logical’, ‘Quant’, ‘Domain’ 
   to values between zero and one using MinMaxScaler().

4) Normalised columns ‘conscientiousness’, ‘agreeableness’, ‘extraversion’, ‘nueroticism’, ‘openess_to_experience’
   to values between -1 and 1 using MinMaxScaler().

5) Normalised column ‘GraduationYear’ to values between 0 and 10 using MinMaxScaler().

5) Split the data into test and train and also shuffle it.

6) label-encoded the output(y) values.

7) Trained the model and checked its accuracy.

At train-test spit = 60-40 , accuracy = 71.81
At train-test spit = 70-30 , accuracy = 72.00
At train-test spit = 75-25 , accuracy = 72.40
At train-test spit = 80-20 , accuracy = 71.00
At train-test spit = 90-10 , accuracy = 71.00

Therefore 75-25 gives the best fit.

"""




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