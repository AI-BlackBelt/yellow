knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
knn.fit(X_train, y_train)

print("test-set score: {:.3f}".format(knn.score(X_test, y_test)))
