from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {'max_features': np.arange(1, X_train.shape[1]+1)}
grid_search = GridSearchCV(RandomForestRegressor(n_estimators=50), param_grid, verbose=3, cv=5, return_train_score=True)
grid_search.fit(X_train, y_train)

scores = pd.DataFrame(grid_search.cv_results_)
scores.groupby("param_max_features").mean().plot(y=["mean_train_score", "mean_test_score"])

imp = pd.DataFrame({"importances": grid_search.best_estimator_.feature_importances_}, index=data.feature_names)
imp = imp.sort_values(by=["importances"], ascending=False)
imp
