df["income_bin"] = df["income"] == " >50K"

df.groupby("age")["income_bin"].mean().plot()
plt.show()

df.groupby("gender")["income_bin"].mean().plot(kind="bar")
plt.show()

df.groupby("education")["income_bin"].mean().plot(kind="bar")
plt.show()

df.groupby("race")["income_bin"].mean().plot(kind="bar")
plt.show()

df.corr()

df.boxplot()
plt.show()

df.groupby("income").hist()
plt.show()

from pandas.plotting import scatter_matrix
scatter_matrix(df, alpha=0.2, figsize=(10, 10), diagonal='kde')
plt.show()
