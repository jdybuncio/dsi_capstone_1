import numpy as np
import pandas as pd
import matplotlib.ply as plt
import sys


%matplotlib inline

def import_student_data():
    # Import data from math and from portuguese students
    math_df = pd.read_csv('../data/student-mat.csv', sep=';')
    port_df = pd.read_csv('../data/student-por.csv', sep=';')
    return math_df, port_df

def check_data_struct(m_df, p_df):
    '''
    Note: these tests are hardcoded based on what I (Taite) have experienced with the data
    '''
    # Get list of column/feature names for each
    m_cols = m_df.columns
    p_cols = p_df.columns

    # Booleans to check for shape of dfs
    m_shape = m_df.shape == (395, 33)
    m_shape_issue = 'shape of Math df'
    # ---> 395 observations, 33 features
    p_shape = p_df.shape == (649, 33)
    p_shape_issue = 'shape of Port df'
    # ---> 649 observations, 33 features

    # Check if the columns are the same across these two dfs
    same_columns = (m_cols == p_cols).all()
    columns_issue = "columns between math/port - don't match up"

    # Check for null values
    m_nulls = m_df.isnull().all().sum() == 0
    m_null_issue = 'nulls in Math df'
    p_nulls = p_df.isnull().all().sum() == 0
    p_null_issue = 'nulls in Port df'
    # ---> Should be none on all of these! (thank god)

    structure_booleans = [m_shape, p_shape, m_nulls, p_nulls, same_columns]
    issue_statement = [m_shape_issue, p_shape_issue, m_null_issue, p_null_issue, columns_issue]

    if structure_booleans.all():
        print("Initial Data looks good. Proceed!")
        return True
    else:
        for idx, cond in enumerate(structure_booleans):
            if not cond:
                print("WARNING: Issue with {}".format(issue_statement[idx]))
        return False

def check_values_counts_across_dfs(cols):
    '''
    Check to make sure we're getting the same number of categories
    for categorical vars and same dummy vars across both math/port dataframes
    '''
    for col in cols:
        # Get list of unique vals in column for MATH df, then PORT df
        m_unique = set(m_df[col].unique())
        p_unique = set(p_df[col].unique())
        # For columns that don't have matching unique values across math/port...
        if m_unique == p_unique or col in ['absences', 'G1', 'G2', 'G3']:
            # Should be true for all except: absences, G1, G2, and G3 (ALL CONTINUOUS, ALL OK)
            # Don't return yet, need to make one more check below...
            continue
        else:
            # If not matching or not one of the cols mentioned above, print and exit script
            print('{0}: {1}, {2}'.format(col, m_unique, p_unique))
            return False

        # Count of unique vals in column for MATH df, then PORT df
        m_count = m_df[col].nunique()
        p_count = p_df[col].nunique()
        # For columns that don't have the same number of unique values across math/port...
        if m_count == p_count or col in ['absences', 'G2', 'G3']:
            # Should be true for all except: absences, G2, and G3 (ALL CONTINUOUS, ALL OK)
            return True
        else:
            # If not matching or not one of the cols mentioned above, print and exit script
            print('{0}: {1}, {2}'.format(col, m_count, p_count))
            return False

def check_col_cats(binary_vars, categorical_vars, appx_cont_vars):
    # Make hardcoded lists of features, by encoding group, based on feature descriptions
    binary = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    categorical = ['Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']
    appx_continuous = ['age', 'absences', 'G1', 'G2', 'G3']

    b = binary_vars == binary
    c = categorical_vars == categorical
    a = appx_cont_vars == appx_continuous

    if all([b, c, a]):
        return True
    else:
        return False

def make_encoding_type_lists(m_df, p_df):
    # Get list of column/feature names for math_df
    # Since we know columns are same, the resulting lists should work for port df too
    cols = m_df.columns
    # Check to make sure unique values are good across the two dfs
    check1 = check_values_counts_across_dfs(cols)
    if check1:
        continue
    else:
        sys.exit("WARNING: issue with matching unique values for columns across math/port!")

    # Make lists for each different type of encoding we want to do...
    binary_vars, categorical_vars, appx_cont_vars = [], [], []

    # Iterating through list of columns, putting each into a category based on how many unique values it has
    for col in cols:
        n_uniq = m_df[col].nunique()
        # Binary / Dummies
        if n_uniq == 2:
            binary_vars.append(col)
        # Categorical (know from looking through feature descriptions)
        elif n_uniq <= 5:
            categorical_vars.append(col)
        # Appx continuous variables
        else:
            appx_cont_vars.append(col)

    # Run hard-coded check on col cats
    check2 = check_col_cats(binary_vars, categorical_vars, appx_cont_vars)
    if check2:
        continue
    else:
        sys.exit("WARNING: issue with column groups for encoding!")
    # Return the lists
    return binary_vars, categorical_vars, appx_cont_vars

def make_dtype_dfs(df, binary_vars, categorical_vars, appx_cont_vars):
    # Function for getting a datatype subset df for each of my main dfs (math and port)
    b_df = df[binary_vars]
    cat_df = df[categorical_vars]
    cont_df = df[appx_cont_vars]
    return b_df, cat_df, cont_df

#Making function so that I can do this easily for both data sets
def encode_dummies(df, binary_vars):
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

    # Remove _d in column names...
    cols = df.columns
    cols = [col[:-2] for col in cols]
    df.columns = cols

    return df, encoding_dict

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
    df.drop(columns=, axis=1, inplace=True)
    return potential_baselines, df




def encode_binary_dummies():
    pass


def join_dfs(cont_df, bin_df, cat_df):
    # JOIN 'EM TOGETHER!
    df = cont_df.join(bin_df.join(cat_df))
    return df

def main():
    # Import both dataframes
    m_df, p_df = import_student_data()

    # Check to make sure initial data import looks good. Stop execution if issue arises.
    initial_data_status = check_data_struct(m_df, p_df)
    if initial_data_status:
        continue
    else:
        sys.exit("ERROR: Issue with initial data import! Fix & rerun to continue.")

    binary_vars, categorical_vars, appx_cont_vars = make_encoding_type_lists(m_df, p_df)

    # Call like this:
    m_bin_df, m_cat_df, m_cont_df = make_dtype_dfs(m_df, binary_vars, categorical_vars, appx_cont_vars)
    p_bin_df, p_cat_df, p_cont_df = make_dtype_dfs(p_df, binary_vars, categorical_vars, appx_cont_vars)

    # Use function to get new encoded dummy dfs and encoding dictionaries
    m_bin_df, m_encoding_dict = encode_dummies(m_bin_df, binary_vars)
    p_bin_df, p_encoding_dict = encode_dummies(p_bin_df, binary_vars)

    # Examine baselines
    m_baselines, m_cat_df = make_dummy_df(m_cat_df, categorical_vars)
    p_baselines, p_cat_df = make_dummy_df(p_cat_df, categorical_vars)

    # JOIN 'EM TOGETHER!
    m_df = join_dfs(m_cont_df, m_bin_df, m_cat_df)
    p_df = join_dfs(p_cont_df, p_bin_df, p_cat_df)






if __name__ == '__main__':
    main()
