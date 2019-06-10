# check https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
print("Precision =", cm[1,1] / (cm[1,1] + cm[0,1]))
print("Recall =", cm[1,1] / (cm[1,1] + cm[1,0]))
