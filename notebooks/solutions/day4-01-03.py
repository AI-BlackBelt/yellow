from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

param_grid = {'n_neighbors': np.arange(1, 15)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, verbose=3, cv=5, return_train_score=True)
grid_search.fit(X_train, y_train)

print("best n_neighbors:", grid_search.best_params_)
