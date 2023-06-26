import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(41)
iris = datasets.load_iris()
X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

navie_bayes_clf = GaussianNB()
navie_bayes_clf = navie_bayes_clf.fit(X_train, Y_train)
Y_predict = navie_bayes_clf.predict(X_test)
score = accuracy_score(Y_test, Y_predict)
print("鳶尾花分類的預測準確率:", score)