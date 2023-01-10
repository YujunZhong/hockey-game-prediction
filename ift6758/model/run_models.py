import os
import sys
import requests
import json
import pickle

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import math 
from numpy import sort
from numpy import mean

from ..viz import *
#from viz.model_plots import plot_roc_curve, plot_goal_rate_cum_goals, calibration_curve
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from dataclasses import dataclass
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from collections import Counter
import imblearn
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.feature_selection import mutual_info_classif
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import StratifiedKFold
from sklearn_evaluation import plot

from comet_ml import Experiment


def process_feats(df):
    """ Convert dataframe to format for training models.
    :param df: input dataframe
    :return: transformed dataframe for training and validation
    """
    # change format into our need
    df['shot_angle'] = df['shot_angle'].abs()

    event_types = {'SHOT': 0, 'GOAL': 1, 'BLOCKED_SHOT': 2, 'CHALLENGE': 3, 'FACEOFF': 4, 'GIVEAWAY': 5, 'HIT': 6, 'MISSED_SHOT': 7, 'PENALTY': 8, 'PERIOD_END': 9,
            'PERIOD_READY': 10, 'PERIOD_START': 11, 'STOP': 12, 'TAKEAWAY': 13, 'PERIOD_OFFICIAL': 14, 'SHOOTOUT_COMPLETE': 15, 'GAME_OFFICIAL': 16, 'GAME_END': 17}
    df['event_type'].replace(event_types, inplace=True)
    df['prev_event_type'].replace(event_types, inplace=True)
    df['prev_event_type'] = pd.to_numeric(df['prev_event_type'])

    shot_types = {'Backhand': 1, 'Deflected': 2, 'Slap Shot': 3, 'Snap Shot': 4, 'Tip-In': 5, 'Wrap-around': 6, 'Wrist Shot': 7}
    df['shot_type'].replace(shot_types, inplace=True)

    bool_to_int = {False: 0, True: 1}
    df['rebound'].replace(bool_to_int, inplace=True)

    df['speed'].replace({np.inf: 0}, inplace=True)

    df['empty_net'].replace(bool_to_int, inplace=True)
    df['empty_net'] = pd.to_numeric(df['empty_net'])

    feat_list = ['game_seconds', 'period', 'coord_x', 'coord_y', 'shot_distance', 'shot_angle', 'shot_type', 'prev_event_type', 'prev_coord_x', 'prev_coord_y', 
            'time_from_prev_event', 'distance_from_prev_event', 'rebound', 'change_in_angle', 'speed', 'empty_net', 'is_home', 'is_forward', 'is_shortHanded', 'event_type']
    for f in feat_list:
        df[f] = df[f].fillna(0)

    return df[feat_list]


def scale_data(X_train, y_train, X_test):
    """ Scale both numerical and categorical variables.
    :param X_train: transformed dataframe for training
    :param y_train: training labels
    :param X_test: transformed dataframe for training (or validation)
    :return: transformed data for training and testing (or validation)
    """
    
    num_features = ['game_seconds', 'coord_x', 'coord_y', 'shot_distance', 'shot_angle', 'prev_coord_x', 'prev_coord_y', 
            'time_from_prev_event', 'distance_from_prev_event', 'change_in_angle', 'speed']
    cat_features = ['period', 'shot_type', 'prev_event_type', 'rebound', 'empty_net', 'is_home', 'is_forward', 'is_shortHanded']
    X_train_categorical = X_train[cat_features]
    X_train_numerical = X_train[num_features]
    
    X_test_categorical = X_test[cat_features]
    X_test_numerical = X_test[num_features]
    
    # Encode shot_type
    encoder=ce.LeaveOneOutEncoder(cols='shot_type', sigma = 0.1)
    loo_res = encoder.fit_transform(X_train_categorical['shot_type'], y_train).rename(columns = {'shot_type': 'shotType'})
    X_train_categorical = pd.concat([loo_res,X_train_categorical], axis =1)
    # Encode prev_event_type
    loo_encoder=ce.LeaveOneOutEncoder(cols='prev_event_type', sigma = 0.1)
    loo_res = loo_encoder.fit_transform(X_train_categorical['prev_event_type'], y_train).rename(columns = {'prev_event_type': 'prevEventType'})
    X_train_categorical = pd.concat([loo_res,X_train_categorical], axis =1)
    # Encode period
    encoder_period=ce.LeaveOneOutEncoder(cols='period', sigma = 0.1)
    loo_res_period = encoder_period.fit_transform(X_train_categorical['period'], y_train).rename(columns = {'period': 'period_encode'})
    X_train_categorical = pd.concat([loo_res_period,X_train_categorical], axis =1)
    
    X_train_categorical = X_train_categorical.drop(columns=['shot_type', 'prev_event_type', 'period'])
    
    X_test_encoder = encoder.transform(X_test_categorical['shot_type']).rename(columns = {'shot_type': 'shotType'})
    X_test_categorical = pd.concat([X_test_encoder,X_test_categorical], axis =1)
    X_test_loo_encoder = loo_encoder.transform(X_test_categorical['prev_event_type']).rename(columns = {'prev_event_type': 'prevEventType'})
    X_test_categorical = pd.concat([X_test_loo_encoder,X_test_categorical], axis =1)
    X_test_loo_encoder = encoder_period.transform(X_test_categorical['period']).rename(columns = {'period': 'period_encode'})
    X_test_categorical = pd.concat([X_test_loo_encoder,X_test_categorical], axis =1)
    X_test_categorical = X_test_categorical.drop(columns=['shot_type', 'prev_event_type', 'period'])
    
    scalar = StandardScaler()
    X_train_numerical = pd.DataFrame(scalar.fit_transform(X_train_numerical), columns = X_train_numerical.columns)
    X_test_numerical = pd.DataFrame(scalar.transform(X_test_numerical), columns = X_test_numerical.columns)

    scaled_X_train = pd.concat([X_train_numerical,X_train_categorical], axis =1)
    scaled_X_test = pd.concat([X_test_numerical,X_test_categorical], axis =1)
    
    return scaled_X_train, scaled_X_test


def fit_pca(X_train, y_train, X_test):
    """ Use PCA to reduce the dimensionality.
    :param X_train: transformed dataframe for training
    :param y_train: training labels
    :param X_test: transformed dataframe for training (or validation)
    :return: transformed data for training and testing (or validation)
    """
    X_train, X_test = scale_data(X_train, y_train, X_test)
    
    pca = PCA(n_components = 0.99)
    pca.fit(X_train)
    reduced_X_train = pca.transform(X_train)
    reduced_X_test = pca.transform(X_test)
    
    return reduced_X_train, reduced_X_test
    
    
def balanced_dataset(X_train, y_train, X_test, y_test, args):

    # define pipeline
    model = get_model(args)
    over = SMOTE(sampling_strategy=0.2, k_neighbors=args.k_neighbors)
    under = RandomUnderSampler(sampling_strategy=0.4)
    steps = [('over', over), ('under', under), ('model', model)]

    pipeline = Pipeline(steps=steps)

    # Fit model
    pipeline.fit(X_train.values, y_train)
    # Make predictions
    preds = pipeline.predict_proba(X_test.values)[:, 1]        
    
    # plot figures
    roc_auc = plot_roc_curve(y_test, preds)
    plot_goal_rate_cum_goals(y_test, preds, args.model)
    calibration_curve(y_test, preds)
    
    return pipeline, roc_auc, preds

    

def feature_selection(X_train, y_train, X_test, y_test, args):
    """ Select features based on different methods.
    :param X_train: transformed dataframe for training
    :param y_train: training labels
    :param X_test: transformed dataframe for training (or validation)
    :param y_test: testing labels
    :param args: contain all the information for the model
    :return: transformed data for training and testing (or validation)
    """
    if args.feature == 'default':
        feat_list = ['game_seconds', 'period', 'coord_x', 'coord_y', 'shot_distance', 'shot_angle', 'shot_type', 'prev_event_type', 'prev_coord_x', 'prev_coord_y', 
            'time_from_prev_event', 'distance_from_prev_event', 'rebound', 'change_in_angle', 'speed', 'empty_net']
        X_train = X_train[feat_list]
        X_test = X_test[feat_list]
        if args.model == "mlp":
            X_train, X_test = scale_data(X_train, y_train, X_test)
            
    elif args.feature == 'extra':
        if args.model == "mlp" or args.model == "hist_gradient_boosting":
            X_train, X_test = scale_data(X_train, y_train, X_test)
    
    elif args.feature == 'importance':
        # Get model
        model = get_model(args)
        sfs = SFS(model,
          k_features = 'best',
          forward= True,
          floating = False,
          verbose= 2,
          scoring= 'roc_auc',
          cv = 5,
          n_jobs= -1
         ).fit(X_train, y_train)
        
        features = list(sfs.k_feature_names_)
        print(features)
        X_train = X_train[features]
        X_test = X_test[features]
    
    elif args.feature == 'mutual_info':
        select_feature = SelectKBest(mutual_info_classif, k=args.num_features)
        select_feature.fit(X_train, y_train)
        features = list(X_train.columns[select_feature.get_support()])
        X_train = X_train[features]
        X_test = X_test[features]

    elif args.feature == 'pca':
        X_train, X_test = fit_pca(X_train, y_train, X_test)

    return np.array(X_train), np.array(X_test)
                                                           

def get_model(args):
    """ Select the model specified by user.
    :param args: contain all the information for the model
    :return: model
    """
    if args.model == "xgboost":
        model = XGBClassifier(booster='dart', eta=0.05, max_depth=10, subsample=0.5)
    elif args.model == "random_forest":
        if args.mode == "best" or args.selection_mode:
            model = RandomForestClassifier(n_estimators=100, max_depth= args.max_depth, min_samples_leaf=args.min_samples_leaf, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif args.model == "cat_boost":  
        if args.mode == "best" or args.selection_mode:
            model = CatBoostClassifier(learning_rate=args.learning_rate, eval_metric='AUC', od_type='IncToDec', verbose=False)
        else:
            model = CatBoostClassifier(eval_metric='AUC', od_type='IncToDec', verbose=False)
    elif args.model == "gradient_boosting":
        if args.mode == "best" or args.selection_mode:
            model = GradientBoostingClassifier(learning_rate=args.learning_rate, n_iter_no_change=5, tol=0.01, random_state=42, verbose=False)
        else:
            model = GradientBoostingClassifier(n_iter_no_change=5, tol=0.01, random_state=42, verbose=False)
    elif args.model == "hist_gradient_boosting": 
        if args.mode == "best" or args.selection_mode:
            model =  HistGradientBoostingClassifier(learning_rate=0.2, max_depth=75, n_iter_no_change=10)
        else:
            model =  HistGradientBoostingClassifier(learning_rate=0.2, max_depth=75, n_iter_no_change=10)
    elif args.model == "mlp":  
        if args.mode == "best" or args.selection_mode:
            model = MLPClassifier(
                activation="logistic",
                solver="adam",
                alpha=args.alpha,
                learning_rate_init=args.learning_rate,
                early_stopping=True,
                max_iter=200,
                hidden_layer_sizes=(args.hidden_layer,args.hidden_layer),
                verbose=False,
            )
        else:
            model = MLPClassifier(
                activation="logistic",
                solver="adam",
                early_stopping=True,
                max_iter=200,
                hidden_layer_sizes=(args.hidden_layer,args.hidden_layer),
                verbose=False,
            )
        
    return model


def train(X_train, y_train, X_test, y_test, args):
    """ training the models and evaluate our models through various methods.
    :param X_train: transformed dataframe for training
    :param y_train: training labels
    :param X_test: transformed dataframe for training (or validation)
    :param y_test: testing labels
    :param args: contain all the information for the model
    :return: 
    """
    roc_auc = 0.0
              
    if args.smoothing:
        model, roc_auc, preds = balanced_dataset(X_train, y_train, X_test, y_test, args)
        
    elif args.mode == "default" or args.mode == "best":
        
        X_train, X_test = feature_selection(X_train, y_train, X_test, y_test, args)
        # Create model instance
        model = get_model(args)     
        # Fit model
        model.fit(X_train, y_train)
        # Make predictions
        preds = model.predict_proba(X_test)[:, 1]

        # plot figures
        roc_auc = plot_roc_curve(y_test, preds)
        plot_goal_rate_cum_goals(y_test, preds, args.model)
        calibration_curve(y_test, preds)
        
    elif args.mode == "tune":
        
        X_train, X_test = feature_selection(X_train, y_train, X_test, y_test, args)
        if args.model == "random_forest":
            param_grid = {
                'max_depth' : [30,40,50],
                'min_samples_leaf': [50, 100, 250],
            }
            
        elif args.model == "cat_boost":
            param_grid = {
                #'learning_rate': np.arange(0.0,0.8,0.2),
                'learning_rate': np.arange(0.16,0.24,0.02),
            }
            
        elif args.model == "gradient_boosting":
            param_grid = {
                'learning_rate': np.arange(0.0,0.8,0.2),
            }
            
        elif args.model == "hist_gradient_boosting":
            param_grid = {
                'learning_rate': np.arange(0.0,0.8,0.2),
                'max_depth' : [30,50,75],
            }
            
        elif args.model == "mlp":
            param_grid = {
                'alphas': np.logspace(-1, 1, 5),
            }
            
        initial_model = get_model(args)

        # Tune hyperparameters
        model = GridSearchCV(estimator=initial_model, param_grid=param_grid, cv=10)
        # Fit model
        model.fit(X_train, y_train)
        print(model.best_params_)
        print(model.best_estimator_)
        print(model.best_score_)

    return model, roc_auc, (y_test, preds)


def log_to_comet(model_path, auc):
    """ log model to the comel.ml project
    :param model_path: the local path where the model is saved
    :param auc: the auc of the model
    :return:
    """
    exp = Experiment(
        api_key=os.environ.get('COMET_API_KEY'), # don’t hardcode!!  'nswhJWNIpDxovklD8fxYpbxX4'
        project_name='ift6758-project',
        workspace='ds-team-9',
    )

    exp.log_model("models", model_path)

    exp.log_metrics({"auc": auc})

