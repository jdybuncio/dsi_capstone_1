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
    # Import data from math and from portuguese students
    math_df = pd.read_csv('data/student-mat.csv', sep=';')
    port_df = pd.read_csv('data/student-por.csv', sep=';')

    # Encode data (categorical and binary variables --> dummies)
    m_df, port_df = encode_data(math_df, port_df)

    # Split data - get holdout and training set
    # For each, make first split "ftrain" ("full train") set, named differently than one-split train/test below
    math_train, math_holdout = train_test_split(m_df, test_size=0.2, random_state=42)
    #math_train, math_test = train_test_split(math_ftrain, test_size=0.2, random_state=42)

    port_train, port_holdout = train_test_split(p_df, test_size=0.2, random_state=42)
    #port_train, port_test = train_test_split(port_ftrain, test_size=0.2, random_state=42)

    # Conduct EDA on training set - mostly just prints/plots things
    conduct_eda(math_train, port_train)

    # Run CV on 3 different models, 3 different thresholds
    best_math, best_port = select_model_logisticCV(math_train, port_train)

    # Use best models for each to get coefficients (train on full train set, test on holdout), and evaluate model via ROC curve and confusion matrix
    # Lots of printing/plotting
    analyze_logistic_regression(best_math, best_port)


if __name__ == '__main__':
    main()
