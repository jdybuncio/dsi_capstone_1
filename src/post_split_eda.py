import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_prep import save_preped_data

def split_for_holdout(df, filename, rand_st=42):
    # Make an initial training/holdout split so we can test how generalizable our model is later
    # Using an 80/20 split
    train_data, holdout_data = train_test_split(df, test_size=0.2, random_state=rand_st)
    save_preped_data(holdout_data, filename + '_holdout')
    save_preped_data(train_data, filename + '_train')
    return train_data, holdout_data

def post_split_eda():
    pass


if __name__ == '__main__':
    m_df = pd.read_csv('../data/prepped_math_data.csv')
    p_df = pd.read_csv('../data/prepped_port_data.csv')

    m_train, m_holdout = split_for_holdout(m_df, 'math', rand_st=42)
    p_train, p_holdout = split_for_holdout(p_df, 'port', rand_st=42)
