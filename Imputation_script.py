import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
import seaborn as sns


df = pd.read_csv('Processed_data_full.csv', index_col=[0])

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



df_KNN = df.copy()

KNN_imputer = KNNImputer(n_neighbors=5)
df_KNN = KNN_imputer.fit_transform(df)
df_KNN = pd.DataFrame(df_KNN)

df_colnames = df.columns
df_KNN.columns = df_colnames

df_knn_bin = df_KNN.copy()
df_knn_bin['birthweight_bin'] = np.where(df_knn_bin['birthweight_g'] < 2500, 1, 0)

df_knn_cat = df_knn_bin.copy()
df_knn_cat['birthweight_cat'] = np.where(df_knn_cat['birthweight_g'] < 1500, 2, df_knn_cat['birthweight_bin'])

df_knn_bin = df_knn_bin.drop(['birthweight_g'], axis=1)
df_knn_cat = df_knn_cat.drop(['birthweight_g', 'birthweight_bin'], axis=1)


df_KNN.to_csv('Data/KNN_data.csv')
df_knn_bin.to_csv('Data/KNN_data_bin.csv')
df_knn_cat.to_csv('Data/KNN_data_cat.csv')