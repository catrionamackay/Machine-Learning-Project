import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Processing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

# Modelling
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error, r2_score,  roc_curve, roc_auc_score, auc, classification_report


df = pd.read_csv('Data/Pipeline_data_bin.csv', index_col=[0])

X = df.drop(['birthweight_bin'], axis=1)
y = df['birthweight_bin']


num_cols = ('bmi','cigs_before_preg','birth_time','m_deliveryweight','f_age','m_age','m_height_in',
            'num_prenatal_visits','prior_births_dead','prior_births_living','prior_terminations','prepreg_weight',
            'num_prev_cesareans','time_since_menses','weight_change','mn_prenatalcare_began')

bin_cols = ('gonorrhea','labour_induced','admit_icu','m_transferred','infections','m_morbidity','riskf',
            'prev_cesarean','infant_sex')

cat_cols = ('birth_attendant','birth_place','birth_mn','birth_dy','f_education','f_hispanic','f_race6',
                'm_nativity','m_education','m_hispanic','m_race6','payment','delivery_method','res_status')


num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first'))])


preprocessor = ColumnTransformer(
        remainder='passthrough', #passthough features not listed
        transformers=[
            ('num', num_transformer , num_cols),
            ('cat', cat_transformer , cat_cols)
        ])


X_processed = preprocessor.fit_transform(X)

X_processed = pd.DataFrame(X_processed)
X_processed.head()


X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=0)


pca = PCA(n_components=50)
pca.fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


lgbm = GridSearchCV(LGBMClassifier(), 
                       {'num_leaves':[1, 5, 10, 30, 50, 70],
                        'learning_rate': [1, 0.5, 0.25, 0.1, 0.05, 0.01],
                        'n_estimators': [32, 50, 64, 75, 100],
                        'max_depths': [1, 2, 3, 5],
                        'min_samples_split': [1, 2, 3, 5, 7],
                        'min_samples_leaf': [0.1, 0.2, 0.5, 1],
                        'max_features': [1, 2, 5, 7, 10]})

lgbm.fit(X_train, y_train)
bestparams = lgbm.best_params_

df_KNN.to_csv('Data/Best_params.csv')

