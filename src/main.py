import data_prep as dp
from post_split_eda import split_train_holdout, post_split_eda

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# import seaborn as sns
# from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
#
# from sklearn.metrics import accuracy_score, classification_report, roc_curve, cross_val_score, train_test_split
# from sklearn import metrics
#
# from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression, Ridge, Lasso
# from sklearn.model_selection import StratifiedKFold, train_test_split, KFold, GridSearchCV, RepeatedStratifiedKFold
#
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import FeatureUnion
#
# from pandas.plotting import scatter_matrix

def split_train_holdout(df):
    pass


def main():
    # Import & Encode data (categorical and binary variables --> dummies)
    m_df, p_df = dp.prep_data()

    print(m_df.head())
    print(p_df.head())

    # Split data - get holdout and training set
    m_train, m_holdout, p_train, p_holdout = split_train_holdout(m_df, p_df)

    # Conduct EDA on training set - mostly just prints/plots things
    #conduct_eda(math_train, port_train)

    # Run CV on 3 different models, 3 different thresholds
    #best_math, best_port = select_model_logisticCV(math_train, port_train)

    # Use best models for each to get coefficients (train on full train set, test on holdout), and evaluate model via ROC curve and confusion matrix
    # Lots of printing/plotting
    #analyze_logistic_regression(best_math, best_port)


if __name__ == '__main__':
    main()
