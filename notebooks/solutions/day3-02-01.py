

# Add a non linear term in the dataframe
df["sin4x"] = np.sin(4*df["x"])

# Build X,y matrices and split them in train/test
X = df[["x", "sin4x"]]
y = df[["y"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Check the coeeficients
print('Weight coefficients: ', regressor.coef_)
print('y-axis intercept: ', regressor.intercept_)

# Compute Prediction
y_pred_train = regressor.predict(X_train)

# Show the prediction on a plot
plt.plot(X_train["x"], y_train, 'o', label="data")
plt.plot(X_train["x"], y_pred_train, 'o', label="prediction")
plt.legend(loc='best')

# Score the model
regressor.score(X_test, y_test)

