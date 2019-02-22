from CV_logistic import select_model_logisticCV
from encoding_cleaning_data import clean_data
from logistic_regression_analysis import analyze_logistic_regression
from post_split_eda import conduct_eda

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.metrics import accuracy_score, classification_report, roc_curve, cross_val_score, train_test_split
from sklearn import metrics

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression, Ridge, Lasso
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold, GridSearchCV, RepeatedStratifiedKFold

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

from pandas.plotting import scatter_matrix

%matplotlib inline




def main():
    pass

if __name__ == '__main__':
    main()
