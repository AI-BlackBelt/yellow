from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

param_grid = {'max_depth': np.arange(1, 15)}
grid_search = GridSearchCV(DecisionTreeRegressor(), param_grid, verbose=3, cv=5, return_train_score=True)
grid_search.fit(X_train, y_train)

print("best max_depth:", grid_search.best_params_)
print("best score:", grid_search.best_score_)
