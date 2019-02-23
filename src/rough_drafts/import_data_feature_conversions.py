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

# Import data from math and from portuguese students
math_df = pd.read_csv('student-mat.csv', sep=';')
port_df = pd.read_csv('student-por.csv', sep=';')

# Get list of column/feature names for each
m_cols = math_df.columns
p_cols = port_df.columns

# Check for shape of df
print(math_df.shape)
# ---> 395 observations, 33 features
print(port_df.shape)
# ---> 649 observations, 33 features

# Check if columns in each are identical
print((m_cols == p_cols).all())

# Check for null values
math_df.info()
port_df.info()
# ---> None! thank god


# Make list of binary features, based on feature descriptions
binary_vars = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
categorical_vars = ['Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']
appx_cont_vars = ['age', 'absences', 'G1', 'G2', 'G3']

# Function for getting a datatype subset df for each of my main dfs (math and port)
def make_dtype_dfs(df, binary_vars, categorical_vars, appx_cont_vars):
    b_df = df[binary_vars]
    cat_df = df[categorical_vars]
    cont_df = df[appx_cont_vars]
    return b_df, cat_df, cont_df

# Call like this:
m_bin_df, m_cat_df, m_cont_df = make_dtype_dfs(math_df, binary_vars, categorical_vars, appx_cont_vars)
p_bin_df, p_cat_df, p_cont_df = make_dtype_dfs(port_df, binary_vars, categorical_vars, appx_cont_vars)

# Check what we've got...
m_bin_df.head()

p_bin_df.head()

print(m_bin_df.shape, m_cat_df.shape, m_cont_df.shape)
print(p_bin_df.shape, p_cat_df.shape, p_cont_df.shape)



#Making function so that I can do this easily for both data sets
def convert_dummies(df, binary_vars):
    encoding_dict = dict()
    for col_name in binary_vars:
        new_col_name = col_name + '_d'
        # Make dictionary for individual col values
        col_dict = dict()
        # Grab unique values for given column
        unique_vals = df[col_name].unique()
        # Sort 'em
        unique_vals.sort()
        # If the first (of two) values is in this list is one of these...
        if unique_vals[0] in ['no', 'F', 'R']:
            # Then we'll encode the second value to equal 1 (first is baseline)
            df[new_col_name] = np.where(df[col_name] == unique_vals[1], 1, 0)
            # Add this info to individual column's disctionary
            col_dict[unique_vals[0]] = 0
            col_dict[unique_vals[1]] = 1
        # Do opposite (switch 1 and 0) for columns with other values
        else:
            df[new_col_name] = np.where(df[col_name] == unique_vals[0], 1, 0)
            col_dict[unique_vals[0]] = 1
            col_dict[unique_vals[1]] = 0
        # Add the column name and its encoding dictionary to the larger dictionary, so we can keep track of baseline values
        encoding_dict[col_name] = col_dict
    df.drop(columns=binary_vars, axis=1, inplace=True)
    return df, encoding_dict


# Use function to get new encoded dummy dfs and encoding dictionaries
m_bin_df, m_encoding_dict = convert_dummies(m_bin_df, binary_vars)
p_bin_df, p_encoding_dict = convert_dummies(p_bin_df, binary_vars)

print(m_bin_df.shape, p_bin_df.shape)
m_bin_df.head()
p_bin_df.head()

# Remove _d in column names...
cols = m_bin_df.columns
cols = [col[:-2] for col in cols]
m_bin_df.columns = cols
p_bin_df.columns = cols

# CHECK WITH:
print(m_bin_df.columns)
print(p_bin_df.columns)

m_bin_df.head()
p_bin_df.head()



# Run this for every binary feature in our dummies dataframe
# for col in binary_vars:
# convert_dummies(m_bin_df, col, m_encoding_dict)
# convert_dummies(p_bin_df, col, p_encoding_dict)

# Drop original category columns (no longer needed)
# m_bin_df.drop(columns=binary_vars, axis=1, inplace=True)
# p_bin_df.drop(columns=binary_vars, axis=1, inplace=True)

# Print resulting encoding dictionary to make sure it's correctly specified
print(m_encoding_dict)
print(p_encoding_dict)
print(m_encoding_dict == p_encoding_dict)



'''
NOTE: NO LONGER GOING TO USE BASELINES IN OG MODEL? PICK BASELINES WHEN PICKING X_COLS...
        ...still have a dictionary of vars that need baselines, w/ potential baseline vars to pick from...
        but will use that to select them later on (not limiting the DataFrame here)
'''
# Make multi-value dummy dataframes
def make_dummy_df(df, categorical_vars):
    # Create empty list that we can insert dummy dataframes into - iterate through to join w/ main df - pick all but last column (One-Hot Encoding)
    dummy_df_lst = []
    potential_baselines = dict()
    # Iterate through list of column names we've picked out
    for col in categorical_vars:
        dummy_df = pd.get_dummies(df[col], prefix=col)
        # Append last column to list of baselines for future reference
        potential_baselines[col] = list(dummy_df.columns)
        dummy_df_lst.append(dummy_df)
    for dummy_df in dummy_df_lst:
        df = df.join(dummy_df)
    # Drop original category columns (no longer needed)
    df.drop(columns=categorical_vars, axis=1, inplace=True)
    return potential_baselines, df

# Examine baselines
m_baselines, m_cat_df = make_dummy_df(m_cat_df, categorical_vars)
p_baselines, p_cat_df = make_dummy_df(p_cat_df, categorical_vars)
print(m_baselines)
print(p_baselines)
print((m_baselines == p_baselines).all())

m_cat_df.head()
p_cat_df.head()

print(m_cat_df.shape)
print(p_cat_df.shape)

# JOIN 'EM TOGETHER!
m_df = m_cont_df.join(m_bin_df.join(m_cat_df))
p_df = p_cont_df.join(p_bin_df.join(p_cat_df))

#Check
m_df.head()
p_df.head()

# Convert test scores into a percentage (out of 20)
for test in ['G1', 'G2', 'G3']:
    m_df[test] = np.where(m_df[test] == 0, 0, m_df[test]/20)
    p_df[test] = np.where(p_df[test] == 0, 0, p_df[test]/20)

# Check
m_df.head()
p_df.head()

'''
NOW things should be ok... Test/Train/Validate split... Then EDA/Viz for each group (pairplot)
'''
