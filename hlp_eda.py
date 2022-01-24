import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import  numpy as np

import scipy.stats as st
from scipy.spatial.distance import squareform

import scipy.special as sp

from scipy.stats import chi2

from tabulate import tabulate


# make my output readable
def pprint_df(dframe):
    print(tabulate(dframe, headers='keys', tablefmt='psql', showindex=True))

# EDA - not main pipeline - pre-pipeline!
# main program in main.py





def check_df(df):

    # let's hunt for missing data

    print('is there a NaN in the table?',df.isnull().values.any())

    # check again
    nan_rows = df[df.isnull().any(1)]
    print('nan rows',nan_rows)

    # check types and examine any non-uniform types (object type, specifically)
    print(df.dtypes)

    # all np.float64
    print(df.head())
    # 0 col looks like index - check with plot linear plot vs df.index

    
    

    # make list of all object columns for deeper inspection
    list_cols = list(df.select_dtypes(['object']))

    # nothing weird, here
    [ print(c,l) for c,l in zip(list_cols,list(map(lambda x: df[x].unique(), list_cols)))]


    # cardinality
    [ print(c,l) for c,l in zip(list_cols,list(map(lambda x: len(df[x].unique()), list_cols)))]

    # we can use OHE for cat vars, but let's look at data distributions to anticipate its effect on RF

    # look at rel freq for categoricals
    [ print(c,l) for c,l in zip(list_cols,list(map(lambda x: df.value_counts([x],normalize=True),list_cols)))]

    # list of variables with only > 0 data



    

   


# only code to execute for runtime proc
try:
    df_app_summary = pd.read_csv('summary_application.csv',sep=',',encoding='utf-8-sig')
except IOError as e:
    print(e)

try:
    df_credit_summary = pd.read_csv('summary_credit_history.csv',sep=',',encoding='utf-8-sig')
except IOError as e:
    print(e)


l_df = [df_app_summary, df_credit_summary]

l_col_bin_encode = ['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY']

l_col_tgt_encode = ['NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE']
    
# roll-up credit_history to just be multinomial count data for each event (0,1,...,5,X,C)
# df_rollup is our multinomial count data target...  
df_rollup = pd.crosstab(df_credit_summary.ID, df_credit_summary.STATUS)

df_rollup.columns = [str(col) + '_count' for col in df_rollup.columns]

print(df_rollup.columns)
# compute sample mean and variance for each column of df_rollup (overdispersion criteria)
print('sample mean values for each client frequency of credit event')
df_ru_means = pd.DataFrame(df_rollup.mean())

df_ru_var = pd.DataFrame(df_rollup.var())

df_merged = df_ru_means.merge(df_ru_var, how='outer', left_index=True, right_index=True)

df_merged.columns = ['sample_mean','sample_variance']


# question: what % id's with at least one 2 has a 5, 3 has a 5, and 4 has a 5?

cols = ['0_count', '1_count', '2_count', '3_count', '4_count', '5_count','C_count','X_count']

print(df_rollup)

# percent rows with at least 1 non-zero 5_count

p_5 = 1-len(df_rollup[(df_rollup['5_count'] == 0)])/len(df_rollup)
print(p_5)

# default (a 5) is extremely rare

p_4 = 1-len(df_rollup[(df_rollup['4_count'] == 0)])/len(df_rollup)
print(p_4)

p_3 = 1-len(df_rollup[(df_rollup['3_count'] == 0)])/len(df_rollup)
print(p_3)

p_2 = 1-len(df_rollup[(df_rollup['2_count'] == 0)])/len(df_rollup)
print(p_2)

p_1 = 1-len(df_rollup[(df_rollup['1_count'] == 0)])/len(df_rollup)
print(p_1)

print(sum((df_rollup['4_count']==0) & (df_rollup['5_count'] > 0)))

print(df_rollup[(df_rollup['4_count']==0) & (df_rollup['5_count'] > 0)])

print(df_rollup[(df_rollup['C_count']==0) & (df_rollup['5_count'] > 0)])

df_rollup['total'] = df_rollup[cols].sum(axis=1)

df_percent = df_rollup.apply(lambda x: round(x/x['total'],3),axis=1)
df_rollup.reset_index(inplace=True)

#print(df_rollup[df_rollup['ID'] == 5011804])
# check df and make basic diagnostic plots
# disabled unless specifically running EDA (this program, not main.py)

# convert from rollup on absolute values per client_id to percentage of rows (months)
# eg 10 5's with 100 c's will be > 1 5, and that's it.



print(df_percent.head())
print(df_credit_summary.head())

# lambda = 0 for G test
# MLE for multinomial distribution of counts

# power_divergence([16, 18, 16, 14, 12, 12], lambda_='log-likelihood')

# compute pairwise divergence between all pairs for each client's count data


# transpose for pairwise row evaluation
# i like this as a utility function for measuring similarity/distance between rows
#df_pwise = df_rollup.T.corr(method=phil_test)


#chi2, p, dof, ex = st.chi2_contingency(df_rollup, lambda_="log-likelihood")
    
#print(df_pwise)


# get distribution of frequency for set of 5_count > 0

df_5_count_pos = df_percent[df_percent['5_count'] > 0]

print(df_5_count_pos.head())

df_5_count_pos['5_count_qcut'] = pd.qcut(df_5_count_pos['5_count'], q=5,
                        labels=['1st',
                                '2nd',
                                '3rd',
                                '4th',
                                '5th'
                                ])

# heuristic - get all freq's above 0.40

df_default_hi = df_5_count_pos[df_5_count_pos['5_count'] > 0.40]
df_default_hi['HIGH_RISK'] = 1

# define 'bad risk' as df_default_hi

df_idx_default = df_default_hi['HIGH_RISK']

df_risk = df_percent.merge(df_idx_default, on='ID',how='left')

df_risk['HIGH_RISK'] = df_risk['HIGH_RISK'].fillna(0)

print(df_risk.head())

if __name__ == '__main__':
    [check_df(i) for i in l_df]

    df_merged.plot(kind='bar')
    plt.title("Sample mean and Variance for each column of count data in credit history\nCounts are aggregated by ID")

    plt.show()



    plt.title("Histogram of accounts with more than 0 5's")

    plt.hist(df_5_count_pos['5_count'],bins=100)
    plt.show()

    plt.title("Histogram of accounts with more 40% 5's")
    plt.hist(df_default_hi['5_count'],bins=10)
    plt.show()
    
    

