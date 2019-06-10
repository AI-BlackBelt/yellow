import pandas as pd
import numpy as np

df = pd.read_csv("data/titanic.csv")
y = df["Survived"]
X = df.drop(["Survived", "PassengerId", "Name"], axis=1)

from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="most_frequent")

X["Age"] = imp.fit_transform(X["Age"].values.reshape(-1, 1)).flatten()
X["Cabin"] = imp.fit_transform(X["Cabin"].values.reshape(-1, 1)).flatten()
X["Embarked"] = imp.fit_transform(X["Embarked"].values.reshape(-1, 1)).flatten()

categories = (X.dtypes == object)

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
tf = make_column_transformer((OneHotEncoder(sparse=False), categories),
                             (StandardScaler(), ~categories),
                             remainder="passthrough")
X_new = tf.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, random_state=0)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=5000)
clf.fit(X_train, y_train)
print("RFs =", clf.score(X_test, y_test))

from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=5000)
clf.fit(X_train, y_train)
print("ETs =", clf.score(X_test, y_test))

from sklearn.svm import SVC
clf = SVC(gamma="scale", C=5.0)
clf.fit(X_train, y_train)
print("SVC =", clf.score(X_test, y_test))
