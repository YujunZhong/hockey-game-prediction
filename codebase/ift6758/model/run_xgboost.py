import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from ..viz import *
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn_evaluation import plot

import os
from comet_ml import Experiment


def proc_feats(df):
    """ Convert dataframe to format for training XGBoost model.
    :param df: input dataframe
    :return: transformed dataframe for training and validation
    """
    df['shot_angle'] = df['shot_angle'].abs()
    event_types = {'SHOT': 0, 'GOAL': 1, 'BLOCKED_SHOT': 2, 'CHALLENGE': 3, 'FACEOFF': 4, 'GIVEAWAY': 5, 'HIT': 6, 'MISSED_SHOT': 7, 'PENALTY': 8, 'PERIOD_END': 9,
            'PERIOD_READY': 10, 'PERIOD_START': 11, 'STOP': 12, 'TAKEAWAY': 13, 'PERIOD_OFFICIAL': 14, 'SHOOTOUT_COMPLETE': 15, 'GAME_OFFICIAL': 16}
    df['event_type'].replace(event_types, inplace=True)
    df['prev_event_type'].replace(event_types, inplace=True)
    df['prev_event_type'] = pd.to_numeric(df['prev_event_type'])

    # set(df.shot_type)
    shot_types = {'Backhand': 1, 'Deflected': 2, 'Slap Shot': 3, 'Snap Shot': 4, 'Tip-In': 5, 'Wrap-around': 6, 'Wrist Shot': 7}
    df['shot_type'].replace(shot_types, inplace=True)

    bool_to_int = {False: 0, True: 1}
    df['rebound'].replace(bool_to_int, inplace=True)

    df['speed'].replace({np.inf: 0}, inplace=True)

    df['empty_net'].replace(bool_to_int, inplace=True)
    df['empty_net'] = pd.to_numeric(df['empty_net'])

    feat_list = ['game_seconds', 'period', 'coord_x', 'coord_y', 'shot_distance', 'shot_angle', 'shot_type', 'prev_event_type', 'prev_coord_x', 'prev_coord_y', 
            'time_from_prev_event', 'distance_from_prev_event', 'rebound', 'change_in_angle', 'speed', 'empty_net']
    for f in feat_list:
        df[f] = df[f].fillna(0)

    return df


def hp_tuning(feats, label):
    """ hyperparameters tuning with GridSearchCV.
    :param feats: input features
    :param label: labels corresponding to features
    :return: instance of fitted estimator
    """
    hyperparams = {'booster': ['gbtree', 'gblinear', 'dart'],
                'objective':['binary:logistic'],
                'eta': [1, 0.1, 0.05],       # learning rate
                'max_depth': [3, 10, 18],
                #   'gamma': [0, 3, 6, 9],
                #   'min_child_weight': [0, 4, 7, 11],
                'subsample': [0.5, 1],
                'eval_metric': ['auc'],
                'seed': [1337]
                }

    xgb_model = XGBClassifier()
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    clf = GridSearchCV(xgb_model, hyperparams, n_jobs=1, 
                    cv=skf,
                    scoring='roc_auc',
                    verbose=2, refit=True)
    clf.fit(feats, label)

    return clf


def hp_tuning_plot(clf):
    """ Visually compare the effects of different hyperparameters.
    :param clf: instance of fitted estimator
    :return:
    """
    # compare hyperparameters with figures
    plot.grid_search(clf.cv_results_, change=('booster', 'eta'),
                    subset={'eval_metric': 'auc', 'max_depth': 10, 'objective': 'binary:logistic', 'seed': 1337, 'subsample': 0.5})

    # plot.grid_search(clf.cv_results_, change=('max_depth', 'subsample'),
    #                  subset={'eval_metric': 'auc','booster': 'dart', 'eta': 0.05, 'objective': 'binary:logistic', 'seed': 1337})

    print('Best AUC score:', clf.best_score_)
    print(clf.best_params_)

    # test_probs = clf.predict_proba(X_val)[:, 1]


def train_xgboost(df, feat_list, xgb_model):
    """ training XGBoost models and evaluate our models through various methods.
    :param train_df: transformed dataframe for training
    :param val_df: transformed dataframe for validation
    :return: 
    """
    label = df[['event_type']]
    if feat_list == ['all']:
        feats = df.drop('event_type', axis=1)
    else:
        feats = df[feat_list]

    X_train, X_val, y_train, y_val = train_test_split(feats, label, test_size=0.1)

    # fit model
    xgb_model.fit(X_train, y_train)

    # make predictions
    preds = xgb_model.predict_proba(X_val)[:, 1]
    labels = y_val['event_type'].to_numpy()

    # plot figures
    roc_auc = plot_roc_curve(labels, preds)
    plot_goal_rate_cum_goals(labels, preds)
    calibration_curve(labels, preds)

    return xgb_model, roc_auc


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


if __name__ == "__main__":
    df = pd.read_csv('../../notebooks/tmp/nhl/nhl_data_clean.csv')
    train_df, val_df = proc_feats(df)

    train_xgboost(train_df, val_df)