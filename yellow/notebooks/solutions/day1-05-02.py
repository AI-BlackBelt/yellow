df = pd.read_csv("data/iris.tsv", sep="\t")
X = df[["sepal-length", "sepal-width", "petal-length", "petal-width"]].values
y = df["target"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
