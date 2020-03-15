import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Processing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold

# Modelling
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
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


ns_probs = [0 for _ in range(len(y_test))]

logm = LogisticRegression(solver='lbfgs', penalty='none')
logm.fit(X_train, y_train)

lr_probs = logm.predict_proba(X_test)

lr_probs = lr_probs[:, 1]

ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)

ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)




knn = neighbors.KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

knn_probs = knn.predict_proba(X_test)

knn_probs = knn_probs[:, 1]

ns_auc = roc_auc_score(y_test, ns_probs)
knn_auc = roc_auc_score(y_test, knn_probs)

ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)



X_train2, X_test2, y_train2, y_test2 = train_test_split(X_processed, y, test_size=0.2, random_state=0)

dt = DecisionTreeClassifier(max_depth=5).fit(X_train2, y_train2)
y_pred2 = dt.predict(X_test2)


dt_probs = dt.predict_proba(X_test2)

dt_probs = dt_probs[:, 1]

ns_auc = roc_auc_score(y_test2, ns_probs)
dt_auc = roc_auc_score(y_test2, dt_probs)

ns_fpr, ns_tpr, _ = roc_curve(y_test2, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(y_test2, dt_probs)


# Plot all roc curves above on one graph

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic Regression', markersize=1)
plt.plot(knn_fpr, knn_tpr, marker='.', label='KNN', markersize=1)
plt.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree', markersize=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('Results/ROC_curves_simple.pdf')
