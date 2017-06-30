import argparse
import sys
import time
from multiprocessing import Process
import pandas as pd
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.model_selection import StratifiedKFold, KFold
import dill

from data import FullDataImporter

data = FullDataImporter()
STEPS = 5
FOLDS = 10
VERBOSE = 1
JOBS = 2

regression_features = ["gpa", "grit", "materialHardship"]
classification_features = ["eviction", "layoff", "jobTraining"]


# run feature selection for different outcomes in parallel
def parallel_feature_selection(func_extract_regression, func_extract_classification):
    processes = []

    for feature in regression_features:
        X, y = data.x_y_for_feature(feature)
        p = Process(target=func_extract_regression, args=(feature, X, y))
        p.start()
        processes.append(p)

    for feature in classification_features:
        X, y = data.x_y_for_feature(feature)
        p = Process(target=func_extract_classification, args=(feature, X, y))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def feature_selection_model():
    parallel_feature_selection(extract_regression_features_model, extract_classification_features_model)


def extract_regression_features_model(feature, X, y):
    # type: (str, pd.DataFrame, pd.Series) -> None
    clf = SelectFromModel(LassoCV(max_iter=20000, cv=FOLDS, n_jobs=JOBS, verbose=VERBOSE))
    clf.fit(X, y)
    print("Selected " + str(sum(clf.get_support())) + " features for " + feature + " using LassoCV.")
    dill.dump(clf.get_support(), open("../featuremasks/model-lasso-" + feature + ".p", "wb"))


def extract_classification_features_model(feature, X, y):
    # type: (str, pd.DataFrame, pd.Series) -> None
    clf = SelectFromModel(LassoCV(max_iter=20000, cv=FOLDS, n_jobs=JOBS, verbose=VERBOSE))
    clf.fit(X, y)
    print("Selected " + str(sum(clf.get_support())) + " features for " + feature + " using LassoCV.")
    dill.dump(clf.get_support(), open("../featuremasks/model-lasso-" + feature + ".p", "wb"))


def feature_selection_en():
    parallel_feature_selection(extract_regression_features_en, extract_classification_features_en)


def extract_regression_features_en(feature, X, y):
    # type: (str, pd.DataFrame, pd.Series) -> None
    clf = SelectFromModel(ElasticNetCV(max_iter=20000, cv=FOLDS, n_jobs=JOBS, verbose=VERBOSE))
    clf.fit(X, y)
    print("Selected " + str(sum(clf.get_support())) + " features for " + feature + " using ElasticNet.")
    dill.dump(clf.get_support(), open("../featuremasks/model-elnet-" + feature + ".p", "wb"))


def extract_classification_features_en(feature, X, y):
    # type: (str, pd.DataFrame, pd.Series) -> None
    clf = SelectFromModel(ElasticNetCV(max_iter=20000, cv=FOLDS, n_jobs=JOBS, verbose=VERBOSE))
    clf.fit(X, y)
    print("Selected " + str(sum(clf.get_support())) + " features for " + feature + " using ElasticNet.")
    dill.dump(clf.get_support(), open("../featuremasks/model-elnet-" + feature + ".p", "wb"))


def feature_selection_rfe():
    parallel_feature_selection(extract_regression_features_rfe, extract_classification_features_rfe)


def extract_regression_features_rfe(feature, X, y):
    # type: (str, pd.DataFrame, pd.Series) -> None
    clf = RFECV(LinearSVR(), step=STEPS, cv=KFold(n_splits=FOLDS, shuffle=True), n_jobs=JOBS, verbose=VERBOSE)
    clf.fit(X, y)
    print("Selected " + str(sum(clf.support_)) + " features for " + feature + " using SVM.")
    dill.dump(clf.get_support(), open("../featuremasks/rfe-svm-" + feature + ".p", "wb"))


def extract_classification_features_rfe(feature, X, y):
    # type: (str, pd.DataFrame, pd.Series) -> None
    clf = RFECV(LinearSVC(), step=STEPS, cv=StratifiedKFold(n_splits=FOLDS, shuffle=True), n_jobs=JOBS, verbose=VERBOSE)
    clf.fit(X, y)
    print("Selected " + str(sum(clf.support_)) + " features for " + feature + " using SVM")
    dill.dump(clf.get_support(), open("../featuremasks/rfe-svm-" + feature + ".p", "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--rfe', action='store_true', default=False, help='Recursive Feature Elimination (SVM)')
    parser.add_argument('--lasso', action='store_true', default=False, help='Select From Model (Lasso)')
    parser.add_argument('--elnet', action='store_true', default=False, help='Select From Model (ElasticNet)')
    args = parser.parse_args()

    sys.stdout = open('../logs/feature-selection-' + str(int(time.time())) + '.log', 'w')

    if not (args.rfe or args.lasso or args.elnet):
        print("You need to specify at least one selection method.")
    if args.rfe:
        print("Starting recursive selection from SVM.")
        feature_selection_rfe()
    if args.lasso:
        print("Starting selection from Lasso.")
        feature_selection_model()
    if args.elnet:
        print("Starting selection from ElasticNet.")
        feature_selection_en()
