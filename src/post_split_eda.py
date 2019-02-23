import data_prep as dp

import numpy as np
import pandas as pd
from sklearn.metrics import train_test_split

def split_for_holdout(m_df, p_df):
    # Make an initial training/holdout split so we can test how generalizable our model is later
    # Using an 80/20 split
    m_train, m_holdout = train_test_split(m_df, test_size=0.2, random_state=42)
    p_train, p_holdout = train_test_split(p_df, test_size=0.2, random_state=42)
    return m_train, m_holdout, p_train, p_holdout

def post_split_eda():
    pass


if __name__ == '__main__':
    post_split_eda()
