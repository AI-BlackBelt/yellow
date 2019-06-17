from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {'n_estimators': [int(i) for i in np.logspace(0, 2, num=10)]}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, verbose=3, cv=5, return_train_score=True)
grid_search.fit(X_train, y_train)

scores = pd.DataFrame(grid_search.cv_results_)
scores.groupby("param_n_estimators").mean().plot(y=["mean_train_score", "mean_test_score"], logx=True)
