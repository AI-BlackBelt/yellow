from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

param_grid = {'n_estimators': [10, 50, 100, 250, 500],
              'learning_rate': [0.5, 0.1, 0.05, 0.01]}
grid_search = GridSearchCV(GradientBoostingRegressor(), param_grid, verbose=3, cv=5, return_train_score=True)
grid_search.fit(X_train, y_train)

scores = pd.DataFrame(grid_search.cv_results_)
scores.groupby("param_n_estimators").mean().plot(y=["mean_train_score", "mean_test_score"])
scores.groupby("param_learning_rate").mean().plot(y=["mean_train_score", "mean_test_score"], logx=True)

print(grid_search.best_score_)
print(grid_search.best_params_)
