import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


data=pd.read_csv('bank-full.csv', header=0, sep=";")
#dataf= data.iloc[:, 0].str.rsplit(';', True)
data=data.dropna()
#print(data.shape)
print(data.shape)
print(list(data.columns))
print(data.head())

#print(data.y[1])
sns.countplot(x='y', data=data, palette='hls')
#plt.show()
plt.clf()

print(data.isnull().sum())

sns.countplot(y="job", data=data)
#plt.show()
plt.clf()

sns.countplot(x="marital", data=data)
#plt.show()
plt.clf()

sns.countplot(x='poutcome', data=data)
#plt.show()
plt.clf()

sns.countplot(x='default', data=data)
#plt.show()
plt.clf()

sns.countplot(x='housing', data=data)
#plt.show()
plt.clf()

#print(data.head())

#print(list(data.columns))
data.drop(data.columns[[0, 5, 3, 8, 9, 10, 11,12, 13, 14]], axis=1, inplace=True)
#print(list(data.columns))

data2 = pd.get_dummies(data, columns = ['job','marital','default','housing','loan','poutcome'])
#data2 = pd.get_dummies(data, columns =['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])
#print(list(data2.columns))

print(list(data2.head()))
data2.drop(data2.columns[[12,16,18,21,24]], axis=1, inplace=True)
print(list(data2.columns))

sns.heatmap(data2.corr())
plt.show()
X = data2.iloc[:,1:]
y = data2.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 0)

print(X_train.shape)
#data2 = pd.get_dummies(data, columns =['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
#print(confusion_matrix())
#print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

print("Accuracy for logistic regression", accuracy_score(y_test, y_pred))


clf_gini = DecisionTreeClassifier(criterion= 'gini', random_state=0, max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

y_pred1 = clf_gini.predict(X_test)

#print('Accuracy of decision tree on test set: {:.2f}'.format(clf_gini.score(X_test, y_test)))
print("Accuracy for decision tree", accuracy_score(y_test, y_pred1))
