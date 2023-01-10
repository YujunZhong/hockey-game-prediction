import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

import shap
from xgboost import XGBClassifier


def univariate(feats, label):
    """ Univariate feature selection method.
    :param feats: input features
    :param label: labels corresponding to features
    :return: selected features
    """
    feats_new = SelectKBest(chi2, k=6).fit_transform(feats, label)

    return feats_new


def recursive(feats, label):
    """ Recursive feature elimination with cross-validation.
    :param feats: input features
    :param label: labels corresponding to features
    :return:
    """
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")

    min_features_to_select = 1  # Minimum number of features to consider
    rfecv = RFECV(
        estimator=svc,
        step=1,
        cv=StratifiedKFold(2),
        scoring="accuracy",
        min_features_to_select=min_features_to_select,
    )
    rfecv.fit(feats.sample(n=500, random_state=1), label.sample(n=500, random_state=1))

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (accuracy)")
    plt.plot(
        range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select),
        rfecv.grid_scores_,
    )
    plt.show()


def l1_method(feats, label):
    """ L1-based feature selection.
    :param feats: input features
    :param label: labels corresponding to features
    :return: selected features
    """
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(feats, label)
    model = SelectFromModel(lsvc, prefit=True)
    feats_new = model.transform(feats)

    return feats_new

def tree_based(feats, label):
    """ Tree-based feature selection.
    :param feats: input features
    :param label: labels corresponding to features
    :return: selected features
    """
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(feats, label)
    clf.feature_importances_  

    model = SelectFromModel(clf, prefit=True)
    feats_new = model.transform(feats)

    return feats_new, model


def shaq(feats, label):
    """ Visualize features by SHAP library.
    :param feats: input features
    :param label: labels corresponding to features
    :return:
    """
    xgb_model = XGBClassifier()

    # fit model
    xgb_model.fit(feats, label)

    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(feats)

    # visualize the first prediction's explanation
    shap.plots.waterfall(shap_values[0])