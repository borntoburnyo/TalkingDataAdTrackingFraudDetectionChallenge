#general import 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

#read data
path = '~/downloads/Dataset/'

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
dtypes = {
    'ip':'uint32',
    'app':'uint16',
    'device':'uint16',
    'os':'uint16',
    'channel':'uint16',
    'click_id':'uint32',
    'is_attributed':'uint8'
}

train = pd.read_csv(path+'train_sample.csv', dtype=dtypes, usecols=train_cols)
test = pd.read_csv(path+'test.csv', dtype=dtypes, usecols=test_cols)
submission = pd.read_csv(path+'sample_submission.csv')

#extract hour and day as new feature 
train['hour'] = pd.to_datetime(train.click_time).dt.hour.astype('uint8')
train['dayofweek'] = pd.to_datetime(train.click_time).dt.dayofweek.astype('uint8')
train.drop(['click_time'], axis=1, inplace=True)

#drop ip as ip doesn't make sense here 
train.drop(['ip'], axis=1, inplace=True)

test['hour'] = pd.to_datetime(test.click_time).dt.hour.astype('uint8')
test['dayofweek'] = pd.to_datetime(test.click_time).dt.dayofweek.astype('uint8')
test.drop(['click_time'], axis=1, inplace=True)

test.drop(['ip'], axis=1, inplace=True)

#train a neural net classifier 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

#define features and target 
features = ['app','device','os','channel','hour','dayofweek']
target = ['is_attributed']

"""
candidate one based on 5-fold CV:

MLPClassifier(hidden_layer_sizes=(150,),
             activation='logistic',
             alpha=2.8e-6,
             learning_rate_init=2.024e-3,
             shuffle=False,
             random_state=0)
"""

#train lightGBM classifier 
import lightgbm as lgb 
from lightgbm.sklearn import LGBMClassifier 

"""
candidate one based on 5-fold CV:

LGBMClassifier(boosting_type='gbdt',
              num_leaves=65,
              max_depth=6,
              learning_rate=0.2,
              n_estimators=1000,
              is_unbalance=True,
              min_child_weight=350,
              min_child_samples=100,
              subsample=0.6,
              colsample_bytree=0.6,
              n_jobs=8,
              random_state=0)
"""

#train xgboost Classifier 
import xgboost as xgb 
from xgboost.sklearn import XGBClassifier

"""
candidate one based on 5-fold CV:

XGBClassifier(max_depth=6,
             learning_rate=0.2,
             n_estimators=500,
             silent=False,
             objective='binary:logistic',
             n_jobs=8,
             min_child_weight=3,
             subsample=0.6,
             colsample_bytree=0.6,
             scale_pos_weight=350,
             random_state=0)
"""

#make meta-learner based on lgb,xgboost and nn
X = train[features].values
y = train[target].values.flatten()

kFold = StratifiedKFold(n_splits = 5, shuffle = False, random_state = 666)
kf = kFold.split(X, y)

clfs = [
    MLPClassifier(hidden_layer_sizes=(150,),
             activation='logistic',
             alpha=2.8e-6,
             learning_rate_init=2.024e-3,
             shuffle=False,
             random_state=0),
    LGBMClassifier(boosting_type='gbdt',
              num_leaves=65,
              max_depth=6,
              learning_rate=0.2,
              n_estimators=1000,
              is_unbalance=True,
              min_child_weight=350,
              min_child_samples=100,
              subsample=0.6,
              colsample_bytree=0.6,
              n_jobs=8,
              random_state=0),
    XGBClassifier(max_depth=6,
             learning_rate=0.2,
             n_estimators=500,
             silent=False,
             objective='binary:logistic',
             n_jobs=8,
             min_child_weight=3,
             subsample=0.6,
             colsample_bytree=0.6,
             scale_pos_weight=350,
             random_state=0)
]

blend_train = np.zeros((train.shape[0], len(clfs)))
blend_test = np.zeros((submission.shape[0], len(clfs)))

for i, clf in enumerate(clfs):
    print("{},{}".format(i+1, clf))
    blend_test_i = np.zeros((submission.shape[0], len(clfs)))
    for j, (train_fold, test_fold) in enumerate(kf):
        print("Fold {}".format(j+1))
        X_train = X[train_fold]
        y_train = y[train_fold]
        X_test = X[test_fold]
        y_test = y[test_fold]
        
        clf.fit(X_train, y_train)
        
        print("Prediction for fold {}, clf {}".format(j+1, i+1))
        y_pred = clf.predict_proba(X_test)[:,1]
        blend_train[test_fold, i] = y_pred
        
        print("Prediction for test set, fold {}, clf {}".format(j+1, i+1))
        blend_test_i[:, j] = clf.predict_proba(test.values)[:,1]
    
    blend_test[:,i] = blend_test_i.mean(axis=1)
    print("Finish clf {}".format(i+1))
print("Finish all stacking.")

#use logistic regression as 2nd stage learner with 1st stage prediction as derived features 
lrc = LogisticRegression()
lrc.fit(blend_train, y)
y_submission = lrc.predict_proba(blend_test)[:, 1]

#try stretching the predictions
y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

submission['is_attributed'] = y_submission

submission.to_csv('stack_sub.csv', index=False)
