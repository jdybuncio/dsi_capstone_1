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


# CONT'D FROM splitting_data.py

'''
    2 full dataframes of encoded data:
        -m_df
        -p_df
'''

# Some plots...

math_rom_cnt = math_ftrain['romantic'].value_counts()
math_rom_perc = math_ftrain['romantic'].value_counts() / math_ftrain.shape[0]
math_rom_perc

math_ftrain['romantic'].value_counts().plot(kind='bar', color=['b', 'r'], alpha=0.5)
port_ftrain['romantic'].value_counts().plot(kind='bar', color=['b', 'r'], alpha=0.5)

sns.countplot(y='G1', hue='romantic', data = math_ftrain)
sns.countplot(y='G2', hue='romantic', data = math_ftrain)
sns.countplot(y='G3', hue='romantic', data = math_ftrain)

sns.countplot(y='G1', hue='romantic', data = port_ftrain)
sns.countplot(y='G2', hue='romantic', data = port_ftrain)
sns.countplot(y='G3', hue='romantic', data = port_ftrain)



# Some important values/stats...

m_rel_status = math_train.groupby('romantic')
p_rel_status = port_train.groupby('romantic')

print(m_rel_status['G1'].mean())
print(p_rel_status['G1'].mean())

m_G1_rel_diff = abs(m_rel_status['G1'].mean()[0] - m_rel_status['G1'].mean()[1])
p_G1_rel_diff = abs(p_rel_status['G1'].mean()[0] - p_rel_status['G1'].mean()[1])

print(m_G1_rel_diff)
print(p_G1_rel_diff)

print(m_rel_status['G1'].std()*3)
print(p_rel_status['G1'].std()*3)

print(m_G1_rel_diff < m_rel_status['G1'].std()*3)
print(p_G1_rel_diff < p_rel_status['G1'].std()*3)

print(m_rel_status['G2'].mean())
print(p_rel_status['G2'].mean())

m_G2_rel_diff = abs(m_rel_status['G2'].mean()[0] - m_rel_status['G2'].mean()[1])
p_G2_rel_diff = abs(p_rel_status['G2'].mean()[0] - p_rel_status['G2'].mean()[1])

print(m_G2_rel_diff)
print(p_G2_rel_diff)


print(m_rel_status['G2'].std()*3)
print(p_rel_status['G2'].std()*3)

print(m_G2_rel_diff < m_rel_status['G2'].std()*3)
print(p_G2_rel_diff < p_rel_status['G2'].std()*3)

print(m_rel_status['G3'].mean())
print(p_rel_status['G3'].mean())

m_G3_rel_diff = abs(m_rel_status['G3'].mean()[0] - m_rel_status['G3'].mean()[1])
p_G3_rel_diff = abs(p_rel_status['G3'].mean()[0] - p_rel_status['G3'].mean()[1])

print(m_G3_rel_diff)
print(p_G3_rel_diff)

print(m_rel_status['G3'].std()*3)
print(p_rel_status['G3'].std()*3)

print(m_G3_rel_diff < m_rel_status['G3'].std()*3)
print(p_G3_rel_diff < p_rel_status['G3'].std()*3)


# sns.pairplot(apprx_cont_df)
#
# sns.pairplot(catvar_df.iloc[:, 6:])
#
# math_df.hist(figsize=(10,10))
# port_df.hist(figsize=(10,10))
#
#
# sns.countplot(y='failures', hue='freetime', data = catvar_df)
#
# sns.countplot(y='G1', hue='romantic', data = math_df)
# sns.countplot(y='G1', hue='romantic', data = port_df)
#
#
# x_corr = full_df_Math[math_predictors].corr()
# plt.matshow(x_corr)
#
# sns.countplot(y='Mjob', hue='Medu', data = math_df)
# sns.countplot(y='Medu', hue='Mjob', data = math_df)
#
# sns.countplot(y='Mjob', data = math_df)
