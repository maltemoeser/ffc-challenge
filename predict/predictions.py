import sys, time
from predictor import Predictor
from data import LassoDataImporter, FullDataImporter, ElasticNetDataImporter, RfeDataImporter
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR, SVC
from sklearn.metrics import brier_score_loss, make_scorer
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor

NJOBS = 24
SEED = 42
CV = 10

def adaboost(data):
    regressor = AdaBoostRegressor(n_estimators=200, random_state=SEED)
    classifier = AdaBoostClassifier(n_estimators=200, random_state=SEED)

    predictor = Predictor(regressor, classifier, data, "adaboost")
    predictor.predict_all()


def random_forrest(data):
    regressor = RandomForestRegressor(n_estimators=200, n_jobs=NJOBS, random_state=SEED)
    classifier = RandomForestClassifier(n_estimators=200, n_jobs=NJOBS, random_state=SEED)

    predictor = Predictor(regressor, classifier, data, "random-forest")
    predictor.predict_all()


def linear(data):
    regressor = ElasticNetCV(cv=CV, random_state=SEED)
    classifier = LogisticRegressionCV(cv=CV, random_state=SEED)

    predictor = Predictor(regressor, classifier, data, "linear")
    predictor.predict_all()


def svm_hyperparameter(data):
    paramsR = [
        {
            "kernel": ["linear"],
            "C": [0.1, 0.2, 0.5, 1, 2, 5],
            "epsilon": [0, 0.1, 0.2, 0.5],
            "tol": [1e-3, 1e-4, 1e-5]
        }, {
            "kernel": ["rbf"],
            "gamma": [0.01, 0.001, 0.0001],
            "C": [0.1, 0.2, 0.5, 1, 2, 5],
            "epsilon": [0, 0.1, 0.2, 0.5],
            "tol": [1e-3, 1e-4, 1e-5]
        }]
    paramsC = [
        {
            "kernel": ["linear"],
            "C": [0.1, 0.2, 0.5, 1, 2, 5],
            "tol": [1e-3, 1e-4, 1e-5]
        }, {
            "kernel": ["rbf"],
            "gamma": [0.01, 0.001, 0.0001],
            "C": [0.1, 0.2, 0.5, 1, 2, 5],
            "tol": [1e-3, 1e-4, 1e-5]
        }]

    regressor = GridSearchCV(SVR(), param_grid=paramsR, scoring="neg_mean_squared_error", n_jobs=NJOBS, cv=CV)
    classifier = GridSearchCV(SVC(probability=True), param_grid=paramsC, scoring=make_scorer(brier_score_loss),
                              n_jobs=NJOBS, cv=CV)

    predictor = Predictor(regressor, classifier, data, "svm")
    predictor.predict_all()


def basic_svm(data):
    regressor = SVR(kernel="linear")
    classifier = SVC(kernel="linear", probability=True)

    predictor = Predictor(regressor, classifier, data, "basic-svm")
    predictor.predict_all()


def gaussian_process(data):
    regressor = GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=10, random_state=SEED)
    classifier = GaussianProcessClassifier(n_restarts_optimizer=10, n_jobs=NJOBS, random_state=SEED)

    predictor = Predictor(regressor, classifier, data, "gp")
    predictor.predict_all()


def run_all():
    d = [LassoDataImporter(), ElasticNetDataImporter(), RfeDataImporter(), FullDataImporter()]

    for data in d:
        random_forrest(data)
        linear(data)
        svm_hyperparameter(data)
        basic_svm(data)
        gaussian_process(data)
        adaboost(data)


if __name__ == '__main__':
    sys.stdout = open('predict-' + str(int(time.time())) + '.log', 'w')
    run_all()
