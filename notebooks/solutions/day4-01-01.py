from sklearn.neighbors import KNeighborsClassifier

cv_scores = []
neighbors = np.arange(1, 15, 2)

for i in neighbors:
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=i), X_train, y_train, cv=5)
    cv_scores.append(np.mean(scores))

best_n_neighbors = neighbors[np.argmax(cv_scores)]
print("best CV score {:.3f}".format(np.max(cv_scores)))
print("best n_neighbors:", best_n_neighbors)
