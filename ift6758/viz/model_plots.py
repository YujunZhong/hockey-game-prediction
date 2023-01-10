import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.calibration import calibration_curve, CalibrationDisplay

sns.set_theme()


def plot_roc_curve(y_true, y_score):
    """ produce a figure using the goal rate as a function of the shot probability model percentile
    :param y_true: data labels in numpy array format
    :param y_score: model predictions in numpy array format
    :return:
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    #roc_auc = auc(fpr, tpr)
    
    roc_auc = roc_auc_score(y_true, y_score)

    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.3f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw = lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.savefig('ROC_curve.png')
    plt.show()

    return roc_auc

def plot_goal_rate_cum_goals(y_true, y_score, label='XGBoost model'):
    """ produce a figure using cumulative proportion of goals (not shots) as a function of the shot probability model percentile.
    :param y_true: data labels in numpy array format
    :param y_score: model predictions in numpy array format
    :return:
    """
    desc_sorted_index = np.argsort(-y_score)
    sorted_label = y_true[desc_sorted_index]
    sorted_score = y_score[desc_sorted_index]

    n = len(y_score)
    bin_num = 20
    bin_centers = list(np.arange(97.5, 0, -100 / bin_num))

    bin_goals = []
    bin_shots = []
    cum_num = 0
    cum_goals = []

    interval = n / bin_num
    for i in range(bin_num):
        cur_bin = sorted_label[int(i * interval): int(min(n, (i+1) * interval))]
        goals = np.sum(cur_bin)
        bin_goals.append(goals)
        bin_shots.append(len(cur_bin))

        cum_num += goals
        cum_goals.append(cum_num)

    df_plot = pd.DataFrame({'bin_centers': bin_centers, 'bin_goals': bin_goals, 'bin_shots': bin_shots, 'cum_goals': cum_goals})
    df_plot['goal_rate'] = df_plot['bin_goals'] / df_plot['bin_shots']
    df_plot['goal_cum_prop'] = df_plot['cum_goals'] / df_plot['bin_goals'].sum()

    ax1 = df_plot.plot(x='bin_centers', y='goal_rate', kind='line', title='Goal rate', xlim=[100, 0], ylim=[0, 1], 
            xlabel='Shot probability model percentile', ylabel='Goals / (Shots + Goals)', label=label)

    ymin, ymax = ax1.get_ylim()
    custom_ticks = np.linspace(ymin, ymax, 11)
    ax1.set_yticks(custom_ticks)
    ax1.set_yticklabels(custom_ticks)

    vals = ax1.get_yticks()
    ax1.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

    xmin, xmax = ax1.get_xlim()
    custom_ticks = np.linspace(xmin, xmax, 11, dtype=int)
    ax1.set_xticks(custom_ticks)
    ax1.set_xticklabels(custom_ticks)
    
    ax1.figure.savefig('Goal_rate.png')

    ax2 = df_plot.plot(x='bin_centers', y='goal_cum_prop', kind='line', title='Cumulative % of goals', xlim=[100, 0], ylim=[0, 1], 
            xlabel='Shot probability model percentile', ylabel='Proportion', label=label)

    ymin, ymax = ax2.get_ylim()
    custom_ticks = np.linspace(ymin, ymax, 11)
    ax2.set_yticks(custom_ticks)
    ax2.set_yticklabels(custom_ticks)

    vals = ax2.get_yticks()
    ax2.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

    xmin, xmax = ax2.get_xlim()
    custom_ticks = np.linspace(xmin, xmax, 11, dtype=int)
    ax2.set_xticks(custom_ticks)
    ax2.set_xticklabels(custom_ticks)
    
    ax2.figure.savefig('Cumulative_goals.png')


def calibration_curve(y_true, y_score):
    """ produce the reliability diagram (calibration curve)
    :param y_true: data labels in numpy array format
    :param y_score: model predictions in numpy array format
    :return:
    """
    # prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10)
    # disp = CalibrationDisplay(prob_true, prob_pred, y_score)
    disp = CalibrationDisplay.from_predictions(y_true, y_score, n_bins=5)
    plt.title("The reliability diagram (calibration curve)")
    
    plt.savefig('calibration_curve.png')
    
#--------- Multiple curve plots ------------#

colors = list(mcolors.TABLEAU_COLORS.keys())
c_size = len(colors)

def plot_roc_curves(results, labels, 
    title = "Receiver operating characteristic (ROC) Curve"):
    """ produce a figure with plots of roc curves of multiple models using their predictions results
    :param results: list(y_true, y_scores)
    :param labels: coressponding list of labels for each of the models results passed
    :param title: title of the figure
    :return:
    """
    lw = 2
    for m in range(len(results)):
        y_true, y_score = results[m]
        curve_label = labels[m]
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw = lw, color = colors[m % c_size],
         label=f'{curve_label} (area = %0.2f)' % roc_auc)
    
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig('Lebel_ROC_Curve_Combine.png')
    plt.show()
    
    
def get_goal_pp(y_true, y_score):
    """ Bins the goals and no-goals based on shot probability model percentile
    :param y_true: list of acutal target values
    :param y_score: list of target prediction probabilities
    :return: dataframe with counts of goals and no-goals for probability percentile bins
    """
    q = np.arange(0, 101, 5)
    bins = np.percentile(y_score, q = q)
    df = pd.DataFrame({'y_true' : y_true, 'y_score': y_score})
    res = df.groupby(['y_true', pd.cut(df['y_score'], 
        bins)]).size().unstack().transpose()
    
    return res

def plot_goal_rate(results, labels, 
    title = 'Goal Rate vs shot probability model percentile'):
    """ produce a figure using goal rates as a function of the shot probability model percentile.
    :param results: list(y_true, y_scores)
    :param labels: coressponding list of labels for each of the models results passed
    :param title: title of the figure
    :return:
    """
    x_points = np.arange(2.5, 100, 5)
    for m in range(len(results)):
        y_true, y_score = results[m]
        curve_label = labels[m]
        goal_pp = get_goal_pp(y_true, y_score)
        goal_rate = (goal_pp[1] / (goal_pp[1] + goal_pp[0])).round(3)

        plt.plot(x_points, np.asarray(goal_rate), label= curve_label, 
            color = colors[m % c_size])
    
    x_ticks = np.arange(0, 101, 5)
    plt.xticks(x_ticks, x_ticks)
    plt.xlim(100, 0)
    plt.ylim(0, 1)
    plt.xlabel("Shot probability model percentile")
    plt.ylabel("Goals / (Shots + Goals)")
    plt.title(title)
    plt.legend()
    plt.savefig('Goal Rate_Combine.png')
    plt.show()
    


def plot_cum_goal_percentages(results, labels, 
    title = 'Cumulative % of goals'):
    """ produce a figure using cumulative proportion of goals (not shots) as a function of the shot probability model percentile.
    :param results: list(y_true, y_scores)
    :param labels: coressponding list of labels for each of the models results passed
    :param title: title of the figure
    :return:
    """
    x_points = np.arange(2.5, 100, 5)
    for m in range(len(results)):
        y_true, y_score = results[m]
        curve_label = labels[m]
        goal_pp = get_goal_pp(y_true, y_score)
        cum_goal = np.cumsum(goal_pp[1].loc[::-1])
        cum_goal_percentage = cum_goal / cum_goal.iloc[-1] * 100

        plt.plot(x_points, np.asarray(cum_goal_percentage.loc[::-1]), label= curve_label, 
            color = colors[m % c_size])
    
    x_ticks = np.arange(0, 101, 5)
    y_ticks = list(range(0, 101, 10))
    y_labels = list(map(lambda x: f'{x}%', y_ticks))
    
    plt.xticks(x_ticks, x_ticks)
    plt.yticks(y_ticks, y_labels)
    plt.xlim(100, 0)
    plt.ylim(0, 100)
    plt.xlabel("Shot probability model percentile")
    plt.ylabel("Proportion")
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig('Lebel_Cumulative_goals_Combine.png')
    plt.show()
    

def plot_calibration_curves(results, labels, 
    title = 'The reliability diagram (calibration curves)', n_bins = 10):
    """ produce the reliability diagram (calibration curves) for each of the models
    :param results: list(y_true, y_scores)
    :param labels: coressponding list of labels for each of the models results passed
    :param title: title of the figure
    :param n_bins: number of bins
    :return:
    """

    ax = plt.gca()
    for m in range(len(results)):
        y_true, y_score = results[m]
        curve_label = labels[m]
        disp = CalibrationDisplay.from_predictions(y_true, y_score, n_bins=n_bins, 
            name = curve_label, color = colors[m % c_size], ax = ax)
    plt.title(title)
    plt.legend(loc="upper right")
    plt.savefig('Lebel_calibration_curves_Combine.png')
    plt.show()
    