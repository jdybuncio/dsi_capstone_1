import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrices
import statsmodels.api as sm
import seaborn as sns
%matplotlib inline

# Import each dataset
math_df = pd.read_csv('student-mat.csv', sep=';')
port_df = pd.read_csv('student-por.csv', sep=';')

# Get list of column/feature names for each
m_cols = math_df.columns
p_cols = port_df.columns

# Check if columns in each are identical
(m_cols == p_cols).all()
# ---> True

cols = list(m_cols)
'''
        ['school',
         'sex',
         'age',
         'address',
         'famsize',
         'Pstatus',
         'Medu',
         'Fedu',
         'Mjob',
         'Fjob',
         'reason',
         'guardian',
         'traveltime',
         'studytime',
         'failures',
         'schoolsup',
         'famsup',
         'paid',
         'activities',
         'nursery',
         'higher',
         'internet',
         'romantic',
         'famrel',
         'freetime',
         'goout',
         'Dalc',
         'Walc',
         'health',
         'absences',
         'G1',
         'G2',
         'G3']
'''

# Remove the variables that differentiate Portuguese/Math students
for col in ['G1', 'G2', 'G3']:
    cols.remove(col)

# Make a merged dataframe of rows with features values that are identical across P/M
# AKA... DataFrame of students who are in BOTH classes
both_df = math_df.merge(port_df, left_on = cols, right_on = cols)
# --> 39 rows (students in both), 36 columns

math_only_df = pd.merge(math_df, both_df, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
print(math_only_df.shape)
math_only_df.head()

port_only_df = pd.merge(port_df, both_df, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
print(port_only_df.shape)
port_only_df.head()

# Create new columns for each... w/ dummy variables
tests = ['G1', 'G2', 'G3']
test_interactions = ['math_G1', 'math_G2', 'math_G3', 'port_G1', 'port_G2', 'port_G3']
dfs = [both_df, math_only_df, port_only_df]
categories = ['both', 'math', 'port']

def make_interaction_test_features(tests, test_interactions, dfs, categories):
    for df in dfs:
        # Make dummy vars for whether or not student is in math, port, and/or both
        for cat in categories:
            if df == both_df:
                # all categories = 1
                df[cat] = 1
                # Set test interactions
                for test in tests:
                    # both_df has both test scores; math marked by '_x' and port by '_y'
                    if cat == 'math':
                        key_term = test + '_x'
                        int_term = cat + '_' + test
                        df[int_term] = df[key_term]
                    elif cat == 'port':
                        key_term = test + '_y'
                        int_term = cat + '_' + test
                        df[int_term] = df[key_term]
            elif (df == math_only_df and cat == 'math') or (df == port_only_df and cat == 'port')
                df[cat] = 1
                for test in tests:
                    int_term = cat + '_' + test
                    df[int_term] = df[test]
            else:
                df[cat] = 0
                for test in tests:
                    int_term = cat + '_' + test
                    df[int_term] = 0

        # Drop columns that are no longer needed
        if df != both_df:
            df.drop(['G1', 'G2', 'G3'], axis=1, inplace=True)
        else:
            df.drop(['G1_x', 'G2_x', 'G3_x', 'G1_y', 'G2_y', 'G3_y'], axis=1, inplace=True)

    # Add unique id for each row
    firstpoint = both_df.shape[0] + 1
    midpoint = port_only_df.shape[0] + firstpoint
    lastpoint = math_only_df.shape[0] + midpoint

    both_df['id'] = range(1, firstpoint)
    port_only_df['id'] = range(firstpoint,  midpoint)
    math_only_df['id'] = range(midpoint, lastpoint)

    joined_df = both_df.append(port_only_df.append(math_only_df, ignore_index=True, sort=False), ignore_index=True, sort=False)

    return




# Check shapes....
print(both_df.shape)
print(port_only_df.shape)
print(math_only_df.shape)



# port_only_df['math'] = 0
# port_only_df['port'] = 1
# port_only_df['both'] = 0
#
# port_only_df['math_G1'] = 0
# port_only_df['port_G1'] = port_only_df['G1']
# port_only_df['math_G2'] = 0
# port_only_df['port_G2'] = port_only_df['G2']
# port_only_df['math_G3'] = 0
# port_only_df['port_G3'] = port_only_df['G3']
#
# math_only_df['math'] = 0
# math_only_df['port'] = 1
# math_only_df['both'] = 0
#
# math_only_df['math_G1'] = math_only_df['G1']
# math_only_df['port_G1'] = 0
# math_only_df['math_G2'] = math_only_df['G2']
# math_only_df['port_G2'] = 0
# math_only_df['math_G3'] = math_only_df['G3']
# math_only_df['port_G3'] = 0
#
# both_df['math'] = 1
# both_df['port'] = 1
# both_df['both'] = 1
#
# both_df['math_G1'] = both_df['G1_x']
# both_df['port_G1'] = both_df['G1_y']
# both_df['math_G2'] = both_df['G2_x']
# both_df['port_G2'] = both_df['G2_y']
# both_df['math_G3'] = both_df['G3_x']
# both_df['port_G3'] = both_df['G3_y']
