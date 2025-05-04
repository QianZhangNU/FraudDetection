import importlib
import feature_generation as FG
importlib.reload(FG)

import feature_engineering as FE
importlib.reload(FE)

import joblib
from sklearn.pipeline import Pipeline

def get_feature_pipeline():
    features = [
        #list of features for target encoding
    ]

    feature_list = [
        #list of selected features
    ]

    feature_pipeline = Pipeline([
        ('fe_shift', FE.shift_difference_transformer([
            #columns of processing
            ])),
        ('fe_diff', FE.generate_difference_transformer(
            #[[pair 1], [ pair 2], ......]
            )),
        ('fe_zscore', FE.generate_zscore_transformer([
            #[[pair 1], [ pair 2], ......]
        ])),
        ('fe_cct', FE.combine_category_transformer([
            #[combined columns 1],
            #[combined columns 2],
            #... ...
        ])),
        ('fe_fmt', FE.Frequency_Multi_Transformer(features, 'label')),
        ('fe_drop_feature', FE.Drop_Feature_Transformer()),
        ('fe_select_feature', FE.Selected_Feature_Transformer(feature_list))
    ])
    
    return feature_pipeline

def load_model():
    XGBoost_Model = joblib.load("xgboost_trained_model.pkl")
    LG_Model = joblib.load("lg_trained_model.pkl")
    RF_Model = joblib.load("rf_trained_model.pkl")
    return XGBoost_Model, LG_Model, RF_Model
