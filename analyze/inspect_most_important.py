import numpy as np
from data import FullDataImporter
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def extract_important(clf, feature):
    print feature

    data = FullDataImporter()

    X, y = data.x_y_for_feature(feature)

    # do training etc.
    clf.fit(X, y)

    # extract importances
    feature_importance = clf.feature_importances_
    n = 5
    # sorting code from https://stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array#23734295
    most_important = np.argpartition(feature_importance, -n)[-n:]
    most_important_sorted = most_important[np.argsort(feature_importance[most_important])][::-1]
    print(X.columns[most_important_sorted], feature_importance[most_important_sorted])


regression_features = ["gpa", "grit", "materialHardship"]
classification_features = ["eviction", "layoff", "jobTraining"]

rfr = RandomForestRegressor(random_state=42)
for feature in regression_features:
    extract_important(rfr, feature)

rfc = RandomForestClassifier(random_state=42)
for feature in classification_features:
    extract_important(rfc, feature)
