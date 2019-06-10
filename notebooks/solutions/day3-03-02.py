
coefs = linreg.coef_
coefsName = boston.feature_names
df_coefs = pd.DataFrame(coefs, index=coefsName, columns=["coef"])
df_coefs["abs"] = df_coefs["coef"].abs()
df_coefs.sort_values("abs", ascending=False)
