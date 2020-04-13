import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer


df = pd.read_csv('Data/Processed_data_full.csv', index_col=[0], low_memory=False)

imputer = IterativeImputer(random_state=0)
df_imputed = pd.DataFrame(imputer.fit_transform(df))

df_colnames = df.columns
df_imputed.columns = df_colnames

df_imputed.to_csv('Data/MICE_data2.csv')
