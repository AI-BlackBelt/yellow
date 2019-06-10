X = X_tfidf
y = df["Score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
regressor = Ridge()
regressor.fit(X_train, y_train)
regressor.score(X_test, y_test)
