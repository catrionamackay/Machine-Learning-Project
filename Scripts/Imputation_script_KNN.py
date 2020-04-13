import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


df = pd.read_csv('Data/Processed_data_full.csv', index_col=[0], low_memory=False)

KNN_imputer = KNNImputer(n_neighbors=5)
df_KNN = pd.DataFrame(KNN_imputer.fit_transform(df))

df_colnames = df.columns
df_KNN.columns = df_colnames

df_KNN.to_csv('Data/KNN_data.csv')
