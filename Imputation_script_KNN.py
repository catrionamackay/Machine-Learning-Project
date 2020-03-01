import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


df = pd.read_csv('Data/Processed_data_full.csv', index_col=[0], low_memory=False)

KNN_imputer = KNNImputer(n_neighbors=5)
df_KNN = pd.DataFrame(KNN_imputer.fit_transform(df))

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
