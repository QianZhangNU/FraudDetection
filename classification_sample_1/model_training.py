# model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import XGBClassifier
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def data_preprocessing(feature_list, target, X_train_s, X_valid_s, X_test_s):
    X_train = X_train_s[feature_list]
    y_train = X_train_s[target]

    X_valid = X_valid_s[feature_list]
    y_valid = X_valid_s[target]

    X_test = X_test_s[feature_list]
    y_test = X_test_s[target]
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def XGB_model_training(X_train, y_train, md=6, lr=0.05, nw=200):
    positive_count = sum(y_train == 1)
    negative_count = sum(y_train == 0)
    scale_pos_weight = negative_count / positive_count

    steps = [("xgb",
             XGBClassifier(scale_pos_weight=scale_pos_weight, max_depth=md,
                          n_jobs=-1, random_state=42, learning_rate=lr,
                          n_estimators=nw, eval_metric='logloss'))]

    model = Pipeline(steps)
    xgb_model = model.fit(X_train, y_train)
    return xgb_model

def LG_model_training(X_train, y_train):
    model = LogisticRegression(random_state=42, class_weight='balanced', C=0.0001)
    lg_model = model.fit(X_train, y_train)
    return lg_model

def RF_model_training(X_train, y_train, nw=200, msl=6):
    rf_model = RandomForestClassifier(random_state=42,
                                    n_jobs=-1,
                                    n_estimators=nw,
                                    min_samples_leaf=msl)
    rf_model.fit(X_train, y_train)
    return rf_model

def permutation_select(model, X_valid, y_valid):
    perm_importance = permutation_importance(model, X_valid, y_valid,
                                           scoring='roc_auc', n_repeats=30, random_state=42)
    feature_name = list(X_valid.columns)
    importance_df = pd.DataFrame({'feature': feature_name,
                                'Importance Mean': perm_importance.importances_mean,
                                'Importance Variance': perm_importance.importances_std}).sort_values(by='Importance Mean',
                                ascending=False)
    feature_list = list(importance_df[importance_df['Importance Mean'] > 0.001]['feature'])
    return importance_df, feature_list

def model_voting_prediction(lg_model, lg_thred, xgb_model, xgb_thred, rf_model, rf_thred, X_test, y_test):
    y_test_proba_lg = lg_model.predict_proba(X_test)
    y_test_pred_lg = (y_test_proba_lg[:, 1] > lg_thred).astype(int)

    y_test_proba_xg = xgb_model.predict_proba(X_test)
    y_test_pred_xg = (y_test_proba_xg[:, 1] > xgb_thred).astype(int)

    y_test_proba_rf = rf_model.predict_proba(X_test)
    y_test_pred_rf = (y_test_proba_rf[:, 1] > rf_thred).astype(int)

    X_test['lg_pred'] = y_test_pred_lg
    X_test['XGB_pred'] = y_test_pred_xg
    X_test['RF_pred'] = y_test_pred_rf

    X_test['Voting_pred'] = 0
    X_test.loc[(X_test['lg_pred']==1) & (X_test['XGB_pred']==1) & (X_test['RF_pred']==1),
              'Voting_pred'] = 1
    y_test_pred = X_test['Voting_pred'].tolist()

    precision = precision_score(y_test, y_test_pred)
    return precision

class hard_voting_model():
    def __init__(self, model_lg, model_xgb, model_rf, thred_lg, thred_xgb, thred_rf):
        self.model_lg = model_lg
        self.model_xgb = model_xgb
        self.model_rf = model_rf
        self.thred_lg = thred_lg
        self.thred_xgb = thred_xgb
        self.thred_rf = thred_rf

    def predict_proba(self, X, y):
        # Ensure only original features are used for prediction
        original_features = [col for col in X.columns if col not in ['lg_pred', 'XGB_pred', 'RF_pred']]
        X_original = X[original_features]

        y_test_proba_lg = self.model_lg.predict_proba(X_original)
        y_test_pred_lg = (y_test_proba_lg[:, 1] > self.thred_lg).astype(int)

        y_test_proba_xg = self.model_xgb.predict_proba(X_original)
        y_test_pred_xg = (y_test_proba_xg[:, 1] > self.thred_xgb).astype(int)

        y_test_proba_rf = self.model_rf.predict_proba(X_original)
        y_test_pred_rf = (y_test_proba_rf[:, 1] > self.thred_rf).astype(int)

        X['lg_pred'] = y_test_pred_lg
        X['XGB_pred'] = y_test_pred_xg
        X['RF_pred'] = y_test_pred_rf

        X['Voting_pred'] = 0
        X.loc[((X['lg_pred']==1) & (X['XGB_pred']==1) & (X['RF_pred']==1)), 'Voting_pred'] = 1
        y_test_pred = X['Voting_pred'].tolist()

        precision = precision_score(y, y_test_pred)
        recall = recall_score(y, y_test_pred)
        f1 = f1_score(y, y_test_pred)
        return precision, recall, f1

class voting_model():
    def __init__(self, model_lg, model_xgb, model_rf):
        self.model_lg = model_lg
        self.model_xgb = model_xgb
        self.model_rf = model_rf

    def predict_proba(self, X):
        y_test_proba_lg = self.model_lg.predict_proba(X)[:, 1]
        y_test_proba_xg = self.model_xgb.predict_proba(X)[:, 1]
        y_test_proba_rf = self.model_rf.predict_proba(X)[:, 1]

        y_voting = (y_test_proba_lg + y_test_proba_xg + y_test_proba_rf) / 3.0
        print('after voting')
        y = np.zeros((len(X), 2))
        y[:, 1] = y_voting
        y[:, 0] = 1 - y_voting

        return y
