import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrices
import statsmodels.api as sm
%matplotlib inline

# Import data from math and from portuguese students
math_df = pd.read_csv('student-mat.csv', sep=';')
port_df = pd.read_csv('student-por.csv', sep=';')

# Check for shape of df
math_df.shape
# ---> 395 observations, 33 features
port_df.shape
# ---> 649 observations, 33 features


# Check for null values
math_df.info()
port_df.info()
# ---> None! thank god


# Get list of column/feature names for each
m_cols = math_df.columns
p_cols = port_df.columns

# Make list of binary features, based on feature descriptions
binary_vars = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

categorical_vars = ['Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']

apprx_continuous_vars = ['age', 'absences', 'G1', 'G2', 'G3']



'''
CONVERT BINARY VALUES TO 1s and 0s
'''

# Doing this for MATH data first!
bin_v_df = math_df[binary_vars]

# Making function so that I can do this easily for PORT data
def convert_dummies(df, col_name, encoding_dict):
    # Make dictionary for individual col values
    col_dict = dict()
    # Grab unique values for given column
    unique_vals = df[col_name].unique()
    # Sort 'em
    unique_vals.sort()
    # If the first (of two) values is in this list is one of these...
    if unique_vals[0] in ['no', 'F', 'R']:
        # Then we'll encode the second value to equal 1 (first is baseline)
        df[col_name] = (df[col_name] == unique_vals[1]).astype(int)
        # Add this info to individual column's disctionary
        col_dict[unique_vals[0]] = 0
        col_dict[unique_vals[1]] = 1
    # Do opposite (switch 1 and 0) for columns with other values
    else:
        df[col_name] = (df[col_name] == unique_vals[0]).astype(int)
        col_dict[unique_vals[0]] = 1
        col_dict[unique_vals[1]] = 0
    # Add the column name and its encoding dictionary to the larger dictionary, so we can keep track of baseline values
    encoding_dict[col_name] = col_dict

# Make a copy, just in case, and then an overarching conversion_dict to hold our encoding info
bv_df = bin_v_df.copy()
encoding_dict = dict()

# Run this for every binary feature in our dummies dataframe
for col in binary_vars:
    convert_dummies(bv_df, col, encoding_dict)

# Print resulting encoding dictionary to make sure it's correctly specified
print(encoding_dict)
# Take a look at the df and check if it's looking right
bv_df.head()


len(binary_vars)
# 13 binary variables
len(categorical_vars)
# 15 categorical variables
len(apprx_continuous_vars)
# 5 approximately continuous variables
len(binary_vars) + len(categorical_vars) +len(apprx_continuous_vars)
# 33 ---> Checks out


# Make df of categorical vars
cv_df = math_df[categorical_vars]
cv_df.head()

# Make multi-value dummy dataframes
def make_dummy_df(df, categorical_vars):
    # Create empty list that we can insert dummy dataframes into - iterate through to join w/ main df - pick all but last column (One-Hot Encoding)
    dummy_df_lst = []
    # Iterate through list of column names we've picked out
    for col in categorical_vars:
        dummy_df = pd.get_dummies(df[col], prefix=col)
        dummy_df_lst.append(dummy_df)

    # Now, iterate through the list of dummy dataframes and join each (EXCEPT for the last column - as per One-Hot Encoding) to the main df
    baselines = []
    for dummy_df in dummy_df_lst:
        # Append last column to list of baselines for future reference
        baselines.append(dummy_df.columns[0])
        df = df.join(dummy_df.iloc[:, 1:])
    df.drop(columns=categorical_vars, axis=1, inplace=True)
    return baselines, df

# Examine baselines
baselines, cv_df = make_dummy_df(cv_df, categorical_vars)
print(baselines)
cv_df.head()
'''
    ['Medu_0', 'Fedu_0', 'Mjob_at_home', 'Fjob_at_home', 'reason_course', 'guardian_father', 'traveltime_1', 'studytime_1', 'failures_0', 'famrel_1', 'freetime_1', 'goout_1', 'Dalc_1', 'Walc_1', 'health_1']
'''

# Make another categorical variable dataframe... this time, a copy!
catvar_df = math_df[categorical_vars].copy()

# Obtain number of dummies req'd, based on unique values and One-Hot encoding
def get_n_one_hot_dummies(df, categorical_vars):
    unique_values, n_dummies = [], []
    for cat in categorical_vars:
        unique_values.append(len(catvar_df[cat].unique()))
        n_dummies.append(len(catvar_df[cat].unique()) - 1)
    total_unique = sum(unique_values)
    total_dummies = sum(n_dummies)
    return total_unique, total_dummies, unique_values, n_dummies


total_unique, total_dummies, unique_values, n_dummies = get_n_one_hot_dummies(catvar_df, categorical_vars)
print("Unique Values: {0}, Total Dummies: {1}".format(total_unique, total_dummies))
# ----> Unique Values: 69, Total Dummies: 54


onehot_dummy_cols = cv_df.columns
'''
    Index(['Medu_1', 'Medu_2', 'Medu_3', 'Medu_4', 'Fedu_1', 'Fedu_2', 'Fedu_3',
           'Fedu_4', 'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher',
           'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher',
           'reason_home', 'reason_other', 'reason_reputation', 'guardian_mother',
           'guardian_other', 'traveltime_2', 'traveltime_3', 'traveltime_4',
           'studytime_2', 'studytime_3', 'studytime_4', 'failures_1', 'failures_2',
           'failures_3', 'famrel_2', 'famrel_3', 'famrel_4', 'famrel_5',
           'freetime_2', 'freetime_3', 'freetime_4', 'freetime_5', 'goout_2',
           'goout_3', 'goout_4', 'goout_5', 'Dalc_2', 'Dalc_3', 'Dalc_4', 'Dalc_5',
           'Walc_2', 'Walc_3', 'Walc_4', 'Walc_5', 'health_2', 'health_3',
           'health_4', 'health_5'],
          dtype='object')
'''

apprx_cont_df = math_df[apprx_continuous_vars].copy()
apprx_cont_df.head()

# Join altogether to get full numerical column
full_df_Math = apprx_cont_df.join(bv_df.join(cv_df))
full_df_Math.head()
# 72 columns = 5 + 13 + 54




'''
LETS DO A PAIR PLOT!
https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166
'''

# Seaborn visualization library
import seaborn as sns
# Create the default pairplot
sns.pairplot(apprx_cont_df)

#catvar_df.iloc[:, 6:]
sns.pairplot(catvar_df.iloc[:, 6:])


'''
LOGISTIC REGRESSION BY MEDIUM:
https://towardsdatascience.com/logistic-regression-classifier-on-census-income-data-e1dbef0b5738
'''
# Initial Model & fitting
math_predictors = ['absences', 'G1', 'G2', 'G3', 'school', 'sex', 'address',
       'schoolsup', 'famsup', 'paid', 'activities',
       'nursery', 'higher']

X = full_df_Math[math_predictors]
y = full_df_Math['Pstatus']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=30)


# Train
X_train = sm.add_constant(X_train)

model = sm.OLS(y_train, X_train)
results = model.fit()

fitted_vals = results.predict(X_train)
stu_resid = results.resid_pearson
residuals = results.resid
y_vals = pd.DataFrame({'residuals':residuals, 'fitted_vals':fitted_vals, 'stu_resid': stu_resid})
print(results.summary())

# Look @ correlation Matrix
x_corr = full_df_Math[math_predictors].corr()
plt.matshow(x_corr)
x_corr

# Residual plot
y_vals.plot(kind='scatter', y='residuals', x='fitted_vals')

# QQ PLOT
fig, ax = plt.subplots(figsize=(12, 5))
fig = sm.qqplot(stu_resid, line='45', fit=True, ax=ax)


# TRAINING --> FIT LOGIT MODEL
model = LogisticRegression(random_state=0)
results = model.fit(X_train, y_train)

fitted_vals = results.predict(X_train)

X_test = sm.add_constant(X_test)
predicted_classes = model.predict(X_test)

# MODEL EVALUATION
acc = accuracy_score(predicted_classes, y_test.values)
print("Accuracy: {}".format(acc))

# PLOT THE CONFUSION Matrix
cfm = confusion_matrix(predicted_classes, y_test.values)
sns.heatmap(cfm, annot=True)
plt.xlabel('Predicted classes')
plt.ylabel('Actual classes')



X_validate = sm.add_constant(X_validate)

predicted_classes = model.predict(X_validate)
# MODEL EVALUATION
accuracy_score(predicted_classes, y_validate.values)
# PLOT THE CONFUSION Matrix
cfm = confusion_matrix(predicted_classes, y_validate.values)
sns.heatmap(cfm, annot=True)
plt.xlabel('Predicted classes')
plt.ylabel('Actual classes')













import numpy as np
import pandas as pd
import io
import requests
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
from pandas.api.types import CategoricalDtype
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
%matplotlib inline

# DATA VIZ
num_attributes.hist(figsize=(10,10))
train_data.describe()

sns.countplot(y='workClass', hue='income', data = cat_attributes)
sns.countplot(y='workClass', hue='income', data = cat_attributes)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


X_cols = make_Xs(df, exclude, 'log_Balance')

y = df['log_Balance'].values
X = df[X_cols].values
X = sm.add_constant(X)
# y = np.log(y)

model = sm.OLS(y, X)
results = model.fit()
fitted_vals = results.predict(X)
stu_resid = results.resid_pearson
residuals = results.resid
y_vals = pd.DataFrame({'residuals':residuals, 'fitted_vals':fitted_vals, 'stu_resid': stu_resid})


print(results.summary())



# TRAINING THE MODEL
train_copy = train_data.copy()
train_copy["income"] = train_copy["income"].apply(lambda x:0 if
                        x=='<=50K' else 1)
X_train = train_copy.drop('income', axis =1)
Y_train = train_copy['income']

X_train_processed=full_pipeline.fit_transform(X_train)
model = LogisticRegression(random_state=0)
model.fit(X_train_processed, Y_train)


# TESTING THE MODEL
test_copy = test_data.copy()
test_copy["income"] = test_copy["income"].apply(lambda x:0 if
                      x=='<=50K.' else 1)
X_test = test_copy.drop('income', axis =1)
Y_test = test_copy['income']

X_test_processed = full_pipeline.fit_transform(X_test)
predicted_classes = model.predict(X_test_processed)

# MODEL EVALUATION
accuracy_score(predicted_classes, Y_test.values)

# PLOT THE CONFUSION Matrix
cfm = confusion_matrix(predicted_classes, Y_test.values)
sns.heatmap(cfm, annot=True)
plt.xlabel('Predicted classes')
plt.ylabel('Actual classes')




# CROSS VALIDATION
cross_val_model = LogisticRegression(random_state=0)
scores = cross_val_score(cross_val_model, X_train_processed,
         Y_train, cv=5)
print(np.mean(scores))


# FINE TUNING

# penalty specifies the norm in the penalization
penalty = ['l1', 'l2']
# C is the inverese of regularization parameter
C = np.logspace(0, 4, 10)
random_state=[0]
# creating a dictionary of hyperparameters
hyperparameters = dict(C=C, penalty=penalty,
                  random_state=random_state)


# GRID SEARCH
clf = GridSearchCV(estimator = model, param_grid = hyperparameters,
                   cv=5)
best_model = clf.fit(X_train_processed, Y_train)
print('Best Penalty:', best_model.best_estimator_.get_params() ['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])


# BEST PARAMS are
Best Penalty: l1
Best C: 1.0


# PREDICTING USING BEST MODEL
best_predicted_values = best_model.predict(X_test_processed)
accuracy_score(best_predicted_values, Y_test.values)

# SAVING THE MODEL
filename = 'final_model.sav'
pickle.dump(model, open(filename, 'wb'))

# LOADING THE MODEL FROM THE PICKLE
saved_model = pickle.load(open(filename, 'rb'))
