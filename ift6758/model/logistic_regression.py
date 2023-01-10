import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ..feature import sep_feature_target
from ..viz import plot_roc_curves, plot_goal_rate, plot_cum_goal_percentages, plot_calibration_curves
def train_lr_with_distance(df: pd.DataFrame):
    """ train logistic regression model using only shot distance information.
    :param df: dataframe with relevant features
    :return: (label, label prediction)
    """
    X, y = sep_feature_target(df, ['shot_distance'])
    return train_lr(X, y)

def train_lr_with_angle(df: pd.DataFrame):
    """ train logistic regression model using only shot angle information.
    :param df: dataframe with relevant features
    :return: (label, label prediction)
    """
    X, y = sep_feature_target(df, ['shot_angle'])
    return train_lr(X, y)

def train_lr_with_dist_and_angle(df: pd.DataFrame):
    """ train logistic regression model using both shot distance and angle information.
    :param df: dataframe with relevant features
    :return: (label, label prediction)
    """
    X, y = sep_feature_target(df, ['shot_distance', 'shot_angle'])
    return train_lr(X, y)

def unifrom_random_predictor(df: pd.DataFrame):
    """ returns the output of a unifrom random predictor.
    :param df: dataframe with relevant features
    :return: (label, label prediction)
    """
    X, y = sep_feature_target(df, ['shot_distance'])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    return (y_val, np.random.uniform(size = len(y_val)))

def train_lr(X, y):
    """ train logistic regression model on the data provided.
    :param X: data contianing feature values
    :param y: target values
    :return: (label, label prediction)
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    
    clf = LogisticRegression(random_state = True, max_iter = 10000, solver='saga')
    clf.fit(X_train, y_train)
    
    predict_prob = clf.predict_proba(X_val)[:,1]
    return clf, (y_val, predict_prob)

def train_base_models(df: pd.DataFrame):
    """ train logistic regression models.
    :param df: dataframe with relevant features
    :return: list of models, list(label, predict_proba).
    """
    res = []
    models = []
    # Train model using only distance data.
    clf_dis, pred = train_lr_with_distance(df)
    models.append(clf_dis)
    res.append(pred)

    # Train model using only angle data
    clf_angle, pred = train_lr_with_angle(df)
    models.append(clf_angle)
    res.append(pred)

    # Train model using distance and angle data.
    clf_dis_angle, pred = train_lr_with_dist_and_angle(df)
    models.append(clf_dis_angle)
    res.append(pred)
    
    # uniform random prediction
    res.append(unifrom_random_predictor(df))

    return models, res

def test_base_models(clfs, df: pd.DataFrame):
    """ Test logistic regression models.
    :param clfs: list of trained Logistic regression models
    :param df: dataframe with relevant features
    :return: list of (label, predict_proba) for each model trained.
    """
    res = []
    
    # Test model using only shot distance data.
    X, y = sep_feature_target(df, ['shot_distance'])
    clf_dis = clfs[0]
    predict_proba = clf_dis.predict_proba(X)[:,1]
    res.append((y, predict_proba))

    # Test model using only shot angle data.
    X, y = sep_feature_target(df, ['shot_angle'])
    clf_angle = clfs[1]
    predict_proba = clf_angle.predict_proba(X)[:,1]
    res.append((y, predict_proba))
    
    # Test model using shot distance and angle data.
    X, y = sep_feature_target(df, ['shot_distance', 'shot_angle'])
    clf_dis_angle = clfs[2]
    predict_proba = clf_dis_angle.predict_proba(X)[:,1]
    res.append((y, predict_proba))

    #unifrom predictor
    predict_proba = np.random.uniform(size = len(y))
    res.append((y, predict_proba))

    return res


def evaluate_base_models(results, 
    labels = ['distance', 'angle', 'distance and angle', 'uniform']):
    """ evaluate the logistic regression models.
    :param results: list(y_true, y_score)
    :return:
    """
    accuracies = compute_accuracies(results, labels)
    print('Accuracies:', accuracies)
    plot_roc_curves(results, labels = labels, title = 'ROC Curve for Logistic Regression')
    plot_goal_rate(results, labels = labels, title = 'Goal rate for Logistic Regression' )
    plot_cum_goal_percentages(results, labels = labels, 
        title = 'Cumulative % of goals for Logistic Regression')
    plot_calibration_curves(results, labels = labels, 
        title = 'The reliability diagram (calibration curves) for Logistic Regression', n_bins = 10)

def compute_accuracies(results, labels):
    """Get the accuracies of all the models.
    :param results: list(y_true, y_scores)
    :param labels:  list of corresponding algo-label for each of the models result
    :return: dictionary of label: accuracy
    """
    acc = []
    for m in range(len(results)):
        y_true, y_score = results[m]
        algo_label = labels[m]
        y_pred = np.where(y_score > 0.5, 1, 0)
        acc.append((algo_label,accuracy_score(y_true, y_pred)))
    return dict(acc)