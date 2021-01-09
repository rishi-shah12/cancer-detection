#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#importing our cancer dataset

dataset = pd.read_csv('data.csv')
dataset = dataset.iloc[:, :-1]
X = dataset.iloc[:, 2:32].values
Y = dataset.iloc[:, 1].values


labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

print(accuracy_score(Y_test, Y_pred)*100)


#print(dataset.head())
#print("Cancer data set dimensions : {}".format(dataset.shape))

#print(dataset.isnull().sum())
#print(dataset.isna().sum())
