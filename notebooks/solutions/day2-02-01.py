df["income_bin"] = df["income"] == " >50K"

df.groupby("age")["income_bin"].mean().plot()
plt.show()

df.groupby("gender")["income_bin"].mean().plot(kind="bar")
plt.show()

df.groupby("education")["income_bin"].mean().plot(kind="bar")
plt.show()

df.groupby("race")["income_bin"].mean().plot(kind="bar")
plt.show()

df.groupby("income").hist(figsize=(8,12))
plt.show()
