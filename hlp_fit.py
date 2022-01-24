# import eda and data prep
import hlp_eda as eda1
import sklearn as sk

from imblearn.ensemble import EasyEnsembleClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV



from sklearn.decomposition import PCA, NMF

from numpy import mean

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

import category_encoders as ce

from category_encoders import TargetEncoder



from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# y_value is HIGH_RISK
df_y = eda1.df_risk['HIGH_RISK']

# X value
df_X = eda1.df_app_summary

df_data = df_X.merge(df_y, on= 'ID')

print(df_data.head())

df_y_final = df_data['HIGH_RISK']

# drop occupation because of NAN's
df_X_final = df_data.drop(['HIGH_RISK','OCCUPATION_TYPE'],1)

print(df_X.shape)

print(df_y.shape)



l_col_tgt_encode = ['NAME_INCOME_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE']

l_col_bin_encode = ['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY']

# implied order, here - works
l_col_ord_encode = ['NAME_EDUCATION_TYPE']

preprocessor = ColumnTransformer(
    transformers=[('ordinal', ce.OrdinalEncoder(), l_col_ord_encode),
                  ('targetbased', ce.LeaveOneOutEncoder(), l_col_tgt_encode),
                  ('ohe', ce.OneHotEncoder(), l_col_bin_encode)
                 ], remainder='passthrough')

#print(preprocessor.fit_transform(df_X_final,df_y_final))
model = EasyEnsembleClassifier()

pipe = Pipeline(steps=[('preprocessor', preprocessor), ('reduce_dim', 'passthrough'),('scaler', 'passthrough'),('classifier', model)])


N_FEATURES_OPTIONS = [2, 4, 8]
C_OPTIONS = [1, 10, 100, 1000]

parameters = [
    {
        # doesn't make sense to use linear decomposition on RF, and so just use it, here
        "reduce_dim": [PCA(iterated_power=7), NMF()],
        "reduce_dim__n_components": N_FEATURES_OPTIONS,

        # DOES make sense to standardize numeric values, though
        "scaler": [StandardScaler()],

        'classifier': (LogisticRegression(class_weight='balanced'),),
        'classifier__C': (0.001,0.01,0.1,1,10,100)
    }, {
        # no sense scaling for RF
        "scaler": ['passthrough'],
        'classifier': (EasyEnsembleClassifier(),),
        'classifier__n_estimators': (10, 30,50)
    }
]
# step 1: compare dim reduction of RF (non-linear) vs PCA (linear)

# need to oversample minority RF - will do this by including ALL minority class(HIGH_RISK) and just bootstrapping majority class


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score)}

gs = GridSearchCV(pipe,param_grid=parameters, n_jobs=-1,cv=cv,scoring=scoring, refit="AUC")
#scores = cross_val_score(pipe, parameters, df_X_final, df_y_final, scoring='roc_auc', cv=cv, n_jobs=-1)

result = gs.fit(df_X_final,df_y_final)

print('Best Score: %s' % result.best_score_)
print('Best hyperparameters: %s ' % result.best_params_)
