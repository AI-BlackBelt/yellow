# Perfect classification (accuracy=1) on easy dataset
from sklearn.linear_model import LogisticRegression
X = np.random.uniform(size=(1000, 3))
X[::2] += 1000
y = X[:, 0] > 500
X_train, X_test, y_train, y_test = train_test_split(X, y)
logreg = LogisticRegression(solver="lbfgs")
logreg.fit(X_train, y_train)
print("score on trivial data: ", logreg.score(X_test, y_test))

# Random classification (accuracy=.5) on random data
y = np.random.normal(size=1000) > .0
X_train, X_test, y_train, y_test = train_test_split(X, y)
logreg = LogisticRegression(solver="lbfgs")
logreg.fit(X_train, y_train)
print("score on random data: ", logreg.score(X_test, y_test))
