from sklearn.dummy import DummyClassifier
dummy = DummyClassifier()
dummy.fit(X_train, y_train)
print(accuracy_score(y_train, dummy.predict(X_train)))
