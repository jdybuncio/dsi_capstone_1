import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

def make_training_Xy_dfs(potential_X_cols, df):
    return (df[X_col] for X_col in potential_X_cols)

def k_fold_CV(X, y, desc, n_folds = 5, cw = 'balanced', threshold = 0.5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=False)
    auc_test = []
    accuracy_test = []
    precision_test = []
    recall_test = []
    f1_score_test = []
    auc_train = []
    accuracy_train = []
    precision_train = []
    recall_train = []
    f1_score_train = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]
        model = LogisticRegression(class_weight=cw)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        probabilities = np.where(probs >= threshold, 1, 0)
        probs_train = model.predict_proba(X_train)[:, 1]
        probabilities_train = np.where(probs_train >= threshold, 1, 0)
        auc_test.append(metrics.roc_auc_score(y_test, probs))
        accuracy_test.append(metrics.accuracy_score(y_test, probabilities))
        precision_test.append(metrics.precision_score(y_test, probabilities))
        recall_test.append(metrics.recall_score(y_test, probabilities))
        f1_score_test.append(metrics.f1_score(y_test, probabilities))
        auc_train.append(metrics.roc_auc_score(y_train, probs_train))
        accuracy_train.append(metrics.accuracy_score(y_train, probabilities_train))
        precision_train.append(metrics.precision_score(y_train, probabilities_train))
        recall_train.append(metrics.recall_score(y_train, probabilities_train))
        f1_score_train.append(metrics.f1_score(y_train, probabilities_train))
    return [np.mean(auc_test), np.mean(accuracy_test), np.mean(precision_test),
            np.mean(recall_test), np.mean(f1_score_test), np.mean(auc_train), np.mean(accuracy_train),
            np.mean(precision_train), np.mean(recall_train), np.mean(f1_score_train), desc]


if __name__ == '__main__':
    m_train = pd.read_csv('../data/prepped_math_train_data.csv')
    p_train = pd.read_csv('../data/prepped_port_train_data.csv')

    # TARGET ALWAYS 'ROMANTIC'
    y_col = 'romantic'
    # FULL MODEL
    X_cols1 = ['age', 'absences', 'G3', 'school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'Medu_1', 'Medu_2', 'Medu_3', 'Medu_4', 'Fedu_1', 'Fedu_2', 'Fedu_3', 'Fedu_4', 'Mjob_at_home', 'Mjob_health', 'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_services', 'Fjob_teacher', 'reason_course', 'reason_home', 'reason_reputation', 'guardian_father', 'guardian_mother', 'traveltime_2', 'traveltime_3', 'traveltime_4', 'studytime_2', 'studytime_3', 'studytime_4', 'failures_1', 'failures_2', 'failures_3', 'famrel_2', 'famrel_3', 'famrel_4', 'famrel_5', 'freetime_2', 'freetime_3', 'freetime_4', 'freetime_5', 'goout_2', 'goout_3', 'goout_4', 'goout_5', 'Dalc_2', 'Dalc_3', 'Dalc_4', 'Dalc_5', 'Walc_2', 'Walc_3', 'Walc_4', 'Walc_5', 'health_2', 'health_3', 'health_4', 'health_5']
    # SMALLER MODEL - TAKING OUT SOME SOCIAL DEMS
    X_cols2 = ['age', 'absences', 'G3', 'school', 'sex', 'address', 'famsize', 'schoolsup', 'activities', 'higher', 'reason_course', 'reason_home', 'reason_reputation', 'studytime_2', 'studytime_3', 'studytime_4', 'failures_1', 'failures_2', 'failures_3', 'freetime_2', 'freetime_3', 'freetime_4', 'freetime_5', 'health_2', 'health_3', 'health_4', 'health_5']
    # SMALLEST MODEL - TAKING OUT EVERYTHING EXCEPT ACADEMIC OUTCOMES
    X_cols3 = ['absences', 'G3', 'activities', 'higher', 'studytime_2', 'studytime_3', 'studytime_4', 'failures_1', 'failures_2', 'failures_3', 'freetime_2', 'freetime_3', 'freetime_4', 'freetime_5', 'health_2', 'health_3', 'health_4', 'health_5']

    potential_X_cols = [X_cols1, X_cols2, X_cols3]

    # Create X and y training dataframes for each group...
    X_m1, X_m2, X_m3 = make_training_Xy_dfs(potential_X_cols, m_train)
    X_p1, X_p2, X_p3 = make_training_Xy_dfs(potential_X_cols, p_train)

    y_m = m_train[y_col]
    y_p = p_train[y_col]

    # Put models into lists so we can iterate through easily...
    m_models = [(X_m1, 'all features'), (X_m2, 'minimal social features'), (X_m3, 'only educ outcome features')]
    p_models = [(X_p1, 'all features'), (X_p2, 'minimal social features'), (X_p3, 'only educ outcome features')]

    m_cv_results = dict()
    p_cv_results = dict()

    for model, m_desc in m_models:
        model_metrics = k_fold_CV(model, y_m, desc=m_desc, n_folds = 5, cw = 'balanced', threshold = 0.5)
        m_cv_results[m_desc] = model_metrics

    for model, p_desc in p_models:
        model_metrics = k_fold_CV(model, y_p, desc=p_desc, n_folds = 5, cw = 'balanced', threshold = 0.5)
        p_cv_results[p_desc] = model_metrics
