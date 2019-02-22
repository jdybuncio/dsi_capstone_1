import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrices
import statsmodels.api as sm
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
%matplotlib inline


# CONT'D FROM import_data_feature_conversions.py

'''
    2 full dataframes of encoded data:
        -m_df
        -p_df
'''

# NOW, SPLIT THE DATAFRAMES INTO TRAIN/TEST/HOLDOUT GROUPS
# For each, make first split "ftrain" ("full train") set, named differently than one-split train/test below
math_ftrain, math_holdout = train_test_split(m_df, test_size=0.2, random_state=42)
math_train, math_test = train_test_split(math_ftrain, test_size=0.2, random_state=42)

port_ftrain, port_holdout = train_test_split(p_df, test_size=0.2, random_state=42)
port_train, port_test = train_test_split(port_ftrain, test_size=0.2, random_state=42)


# to check:
print(math_train.shape)
print(port_train.shape)

print(math_test.shape)
print(port_test.shape)

print(math_holdout.shape)
print(port_holdout.shape)

print(math_train.shape[0] + math_test.shape[0] + math_holdout.shape[0])
print(port_train.shape[0] + port_test.shape[0] + port_holdout.shape[0])


# Look through baselines...
for key in m_baselines.keys():
    print(key, m_baselines[key])

# Set baselines
baselines = ['Medu_0', 'Fedu_0', 'Mjob_other', 'Fjob_other', 'reason_other', 'guardian_other', 'traveltime_1', 'studytime_1', 'failures_0', 'famrel_1', 'freetime_1', 'goout_1', 'Dalc_1', 'Walc_1', 'health_1']


# Start with OLS regression -->


'''
    NEW MODEL:
        Y = romantic, Pstatus, famsup
        X_dem = age, sex, school, address, famsize
        X_edu_char = freetime, schoolsup
        X_ed_outcomes = G1/G2/G3, failures, absences, higher, studytime, activities, reason

    REMOVE:
        cats: health

'''

cats_to_remove = ['Medu', 'Fedu', 'Mjob', 'Fjob', 'guardian', 'traveltime', 'famrel', 'goout', 'Dalc', 'Walc', 'health']

X_cols = ['age', 'absences', 'G3', 'school', 'sex', 'address', 'famsize', 'schoolsup', 'activities', 'higher', 'reason_course', 'reason_home', 'reason_reputation', 'studytime_2', 'studytime_3', 'studytime_4', 'failures_1', 'failures_2', 'failures_3', 'freetime_2', 'freetime_3', 'freetime_4', 'freetime_5', 'health_2', 'health_3', 'health_4', 'health_5']

targets = ['romantic', 'Pstatus', 'famsup']





# Let's run OLS to see which characteristics are significant (regarding test scores)

X_cols_OLS = ['age', 'absences', 'G1', 'G2', 'school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'Medu_1', 'Medu_2', 'Medu_3', 'Medu_4', 'Fedu_1', 'Fedu_2', 'Fedu_3', 'Fedu_4', 'Mjob_at_home', 'Mjob_health', 'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_services', 'Fjob_teacher', 'reason_course', 'reason_home', 'reason_reputation', 'guardian_father', 'guardian_mother', 'traveltime_2', 'traveltime_3', 'traveltime_4', 'studytime_2', 'studytime_3', 'studytime_4', 'failures_1', 'failures_2', 'failures_3', 'famrel_2', 'famrel_3', 'famrel_4', 'famrel_5', 'freetime_2', 'freetime_3', 'freetime_4', 'freetime_5', 'goout_2', 'goout_3', 'goout_4', 'goout_5', 'Dalc_2', 'Dalc_3', 'Dalc_4', 'Dalc_5', 'Walc_2', 'Walc_3', 'Walc_4', 'Walc_5', 'health_2', 'health_3', 'health_4', 'health_5']
y_col = 'G3'

X_train_OLS['constant'] = 1

y_col = 'romantic'

# FULL MODEL
X_cols1 = ['age', 'absences', 'G3', 'school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'Medu_1', 'Medu_2', 'Medu_3', 'Medu_4', 'Fedu_1', 'Fedu_2', 'Fedu_3', 'Fedu_4', 'Mjob_at_home', 'Mjob_health', 'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_services', 'Fjob_teacher', 'reason_course', 'reason_home', 'reason_reputation', 'guardian_father', 'guardian_mother', 'traveltime_2', 'traveltime_3', 'traveltime_4', 'studytime_2', 'studytime_3', 'studytime_4', 'failures_1', 'failures_2', 'failures_3', 'famrel_2', 'famrel_3', 'famrel_4', 'famrel_5', 'freetime_2', 'freetime_3', 'freetime_4', 'freetime_5', 'goout_2', 'goout_3', 'goout_4', 'goout_5', 'Dalc_2', 'Dalc_3', 'Dalc_4', 'Dalc_5', 'Walc_2', 'Walc_3', 'Walc_4', 'Walc_5', 'health_2', 'health_3', 'health_4', 'health_5']


# SMALLER MODEL - TAKING OUT SOME SOCIAL DEMS
X_cols2 = ['age', 'absences', 'G3', 'school', 'sex', 'address', 'famsize', 'schoolsup', 'activities', 'higher', 'reason_course', 'reason_home', 'reason_reputation', 'studytime_2', 'studytime_3', 'studytime_4', 'failures_1', 'failures_2', 'failures_3', 'freetime_2', 'freetime_3', 'freetime_4', 'freetime_5', 'health_2', 'health_3', 'health_4', 'health_5']

# SMALLEST MODEL - TAKING OUT EVERYTHING EXCEPT ACADEMIC OUTCOMES
X_cols3 = ['absences', 'G3', 'activities', 'higher', 'studytime_2', 'studytime_3', 'studytime_4', 'failures_1', 'failures_2', 'failures_3', 'freetime_2', 'freetime_3', 'freetime_4', 'freetime_5', 'health_2', 'health_3', 'health_4', 'health_5']







full_X_train_cols = X_train_cols.copy()

# Need to specify features to use as predictors... Take out some dummies for One-Hot Encoding, as well as taking out G1/2
X_train_cols.remove('G1')
X_train_cols.remove('G2')

# Remove target from X_train columns
# Could generalize code more w/ .pop() function?
X_train_cols.remove('romantic')
y_train_col = 'romantic'

# Check
print(X_train_cols)
print(y_train_col)


# Create X and y training dataframes for each group...
X_m = math_ftrain[X_train_cols]
X_p = port_ftrain[X_train_cols]

y_m = math_ftrain[y_train_col]
y_p = port_ftrain[y_train_col]

X_m.head()


model = sm.OLS(y_train_OLS, X_train_OLS)
results = model.fit()

fitted_vals = results.predict(X_train_OLS)
stu_resid = results.resid_pearson
residuals = results.resid
#y_vals = pd.DataFrame({'residuals':residuals, 'fitted_vals':fitted_vals, 'stu_resid': stu_resid})
print(results.summary())




# Train
#X_train_OLS = sm.add_constant(X_train_OLS)
X_train_OLS['constant'] = 1


model = sm.OLS(y_train_OLS, X_train_OLS)
results = model.fit()

fitted_vals = results.predict(X_train_OLS)
stu_resid = results.resid_pearson
residuals = results.resid
#y_vals = pd.DataFrame({'residuals':residuals, 'fitted_vals':fitted_vals, 'stu_resid': stu_resid})
print(results.summary())
