import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer


df = pd.read_csv('Data/Processed_data_full.csv', index_col=[0], low_memory=False)

df = df.drop(['marital_stat'], axis=1)


imputer = IterativeImputer(random_state=0)
df_imputed = pd.DataFrame(imputer.fit_transform(df))

df_colnames = df.columns
df_imputed.columns = df_colnames

df_bin = df_imputed.copy()
df_bin['birthweight_bin'] = np.where(df_bin['birthweight_g'] < 2500, 1, 0)

df_cat = df_bin.copy()
df_cat['birthweight_cat'] = np.where(df_cat['birthweight_g'] < 1500, 2, df_cat['birthweight_bin'])

df_bin = df_bin.drop(['birthweight_g'], axis=1)
df_cat = df_cat.drop(['birthweight_g', 'birthweight_bin'], axis=1)

df_imputed.to_csv('Data/MICE_data.csv')
df_bin.to_csv('Data/MICE_data_bin.csv')
df_cat.to_csv('Data/MICE_data_cat.csv')
