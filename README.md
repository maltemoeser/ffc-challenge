# Fragile Families Challenge in Python

## Setup

- Clone this repository
- Create a new virtual python environment: `virtualenv --python=/usr/bin/python2.7 venv`
- Activate the virtualenv: `source venv/bin/activate`
- Install requirements: `pip install -r requirements.txt`

## Data

To receive the data set you will need to [apply for the FFC challenge](http://www.fragilefamilieschallenge.org/apply/) and agree to their terms of service.
Note that the terms of service forbid us to provide you with the data directly.

## Data imputation

You can use the `imputation/impute.R` script to create an imputed version of the FFC data.
Note that removing highly correlated columns will require at least 8GB of free memory, and might remove columns that are of interest to you if you follow a *social scientists* approach.

Make sure to check the FFC website for imputation scripts in other languages.

## Prediction

### Feature selection

The code provides multiple options for feature selection, including Lasso and Elastic Net regression as well as recursive feature elimination.
We provide the necessary boolean masks in the folder `featuremasks`, but you are free to create your own.
Check out `data.py` for a convenient way to load the data.

### Classifiers and Regressors

You can modify existing classification and regression methods or add your own in `predictions.py`.

To use XGBoost, follow the installation instructions [here](https://xgboost.readthedocs.io/en/latest/build.html#building-on-ubuntu-debian) instead of installing it with pip.

