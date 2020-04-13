import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

df = pd.read_csv('Data/Processed_data_full.csv', index_col=[0], low_memory=False)

numeric_cols = ('bmi','cigs_before_preg','birthweight_g','birth_time','m_deliveryweight','f_age','m_age',
               'm_height_in','num_prenatal_visits','prior_births_dead','prior_births_living','prior_terminations',
               'prepreg_weight','num_prev_cesareans','time_since_menses')

cat_cols = ('birth_attendant','birth_place','birth_mn','birth_dy','f_education','f_hispanic','f_race6','gonorrhea',
           'labour_induced','m_nativity','m_education','m_hispanic','admit_icu','m_race6','m_transferred',
           'infections','m_morbidity','riskf','payment','mn_prenatalcare_began','delivery_method','res_status',
           'prev_cesarean','infant_sex')

numeric_transformer = Pipeline(steps=[
    ('imputer', IterativeImputer(random_state=0))])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))])

preprocessor = ColumnTransformer(transformers=[
      ('num', numeric_transformer, numeric_cols),
      ('cat', cat_transformer, cat_cols)])

df_processed = preprocessor.fit_transform(df)

df_processed = pd.DataFrame(df_processed)
df_processed.columns = numeric_cols + cat_cols

df_processed['weight_change'] = df_processed['m_deliveryweight'] - df_processed['prepreg_weight']

df_bin = df_processed.copy()
df_bin['birthweight_bin'] = np.where(df_bin['birthweight_g'] < 2500, 1, 0)

df_cat = df_bin.copy()
df_cat['birthweight_cat'] = np.where(df_cat['birthweight_g'] < 1500, 2, df_cat['birthweight_bin'])

df_bin = df_bin.drop(['birthweight_g'], axis=1)
df_cat = df_cat.drop(['birthweight_g', 'birthweight_bin'], axis=1)


df_processed.to_csv('Data/Pipeline_data.csv')
df_bin.to_csv('Data/Pipeline_data_bin.csv')
df_cat.to_csv('Data/Pipeline_data_cat.csv')
