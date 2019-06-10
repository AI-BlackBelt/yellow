dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)

fpr, tpr, thresholds = roc_curve(y_train, dt.predict_proba(X_train)[:, 1])
plt.plot(fpr, tpr)
plt.show()

print("ROC AUC =", roc_auc_score(y_train, dt.predict_proba(X_train)[:, 1]))
