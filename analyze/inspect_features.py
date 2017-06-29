from itertools import compress
import dill
from data import FullDataImporter

data = FullDataImporter()

features = data.all_features()

print(list(compress(features.columns, dill.load(open("../featuremasks/rfe-svm-grit.p")))))
print(list(compress(features.columns, dill.load(open("../featuremasks/rfe-svm-layoff.p")))))
print(list(compress(features.columns, dill.load(open("../featuremasks/rfe-svm-jobTraining.p")))))
print(list(compress(features.columns, dill.load(open("../featuremasks/model-elasticnet-jobTraining.p")))))
print(list(compress(features.columns, dill.load(open("../featuremasks/model-lasso-jobTraining.p")))))

print(sum(dill.load(open("../featuremasks/model-elasticnet-gpa.p"))))
print(sum(dill.load(open("../featuremasks/model-elasticnet-grit.p"))))
print(sum(dill.load(open("../featuremasks/model-elasticnet-materialHardship.p"))))
print(sum(dill.load(open("../featuremasks/model-elasticnet-eviction.p"))))
print(sum(dill.load(open("../featuremasks/model-elasticnet-layoff.p"))))
print(sum(dill.load(open("../featuremasks/model-elasticnet-jobTraining.p"))))

print(sum(dill.load(open("../featuremasks/model-lasso-gpa.p"))))
print(sum(dill.load(open("../featuremasks/model-lasso-grit.p"))))
print(sum(dill.load(open("../featuremasks/model-lasso-materialHardship.p"))))
print(sum(dill.load(open("../featuremasks/model-lasso-eviction.p"))))
print(sum(dill.load(open("../featuremasks/model-lasso-layoff.p"))))
print(sum(dill.load(open("../featuremasks/model-lasso-jobTraining.p"))))

print(sum(dill.load(open("../featuremasks/rfe-svm-gpa.p"))))
print(sum(dill.load(open("../featuremasks/rfe-svm-grit.p"))))
print(sum(dill.load(open("../featuremasks/rfe-svm-materialHardship.p"))))
print(sum(dill.load(open("../featuremasks/rfe-svm-eviction.p"))))
print(sum(dill.load(open("../featuremasks/rfe-svm-layoff.p"))))
print(sum(dill.load(open("../featuremasks/rfe-svm-jobTraining.p"))))
