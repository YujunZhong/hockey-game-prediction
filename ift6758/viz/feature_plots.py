import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def hist_dist_shot_counts(df: pd.DataFrame):
	""" produces a histogram plot of no. of goals and shots w.r.t distances.
	:param df: dataframe with relevant features
	:return:
	"""
	#prepare data
	goals = df.loc[df['event_type'] == 'GOAL']['shot_distance']
	no_goals = df.loc[df['event_type'] == 'SHOT']['shot_distance']
	bins = np.arange(0, 201, 10)
	
	#plot
	plt.style.use('seaborn-deep')	
	plt.figure(figsize = (10, 5))
	plt.hist([no_goals, goals], bins = bins, rwidth = .95, label = ['no-goal', 'goal'])
	plt.yscale('log')
	plt.ylim((1, 10**5))
	plt.legend(loc='upper right')
	plt.title("Shot Distance to Goal/No-Goal")
	plt.xlabel('distance (ft.)')
	plt.ylabel('count')
	plt.xticks(bins, bins)
	plt.show()



def hist_angle_shot_counts(df: pd.DataFrame):
	""" produces a histogram plot of no. of goals and shots w.r.t angles.
	:param df: dataframe with relevant features
	:return:
	"""
	#prepare data
	goals = df.loc[df['event_type'] == 'GOAL']['shot_angle']
	no_goals = df.loc[df['event_type'] == 'SHOT']['shot_angle']
	bins = np.arange(-90, 91, 10)
	
	#plot
	plt.style.use('seaborn-deep')
	plt.figure(figsize = (10, 5))
	plt.hist([no_goals, goals], bins = bins, rwidth = .95, label = ['no-goal', 'goal'])
	plt.yscale('log')
	plt.ylim((1, 10**5))
	plt.legend(loc='upper right')
	plt.title("Shot Angle to Goal/No-Goal")
	plt.xlabel('angle (deg.)')
	plt.ylabel('count')
	plt.xticks(bins, bins)
	plt.show()

def hist_dist_angle_shot_counts(df: pd.DataFrame):
	""" produces a 2d histogram plot of goal rate w.r.t angle and distance.
	:param df: dataframe with relevant features
	:return:
	"""
	#prepare data
	goals = df.loc[df['event_type'] == 'GOAL'][['shot_distance', 'shot_angle']]
	no_goals = df.loc[df['event_type'] == 'SHOT'][['shot_distance', 'shot_angle']]
	dist_bins = np.arange(0, 201, 10)
	angle_bins = np.arange(-90, 91, 10)

	#plot 1
	g = sns.jointplot(data=goals, x='shot_distance', y='shot_angle', 
		height=9, ratio=10, kind="hist", cbar=True, xlim=[0,200], ylim=[-90,90],
		color = '#66a182', joint_kws = dict(bins=(dist_bins, angle_bins)),
		marginal_ticks = True)
	g.ax_joint.set_xticks(dist_bins)
	g.ax_joint.set_yticks(angle_bins)
	g.ax_joint.set_xlabel('distance (ft.)')
	g.ax_joint.set_ylabel('angle (deg.)')
	g.fig.suptitle('Distance and Angle to Goals')
	g.fig.tight_layout()

	#plot 2
	g = sns.jointplot(data=no_goals, x='shot_distance', y='shot_angle', 
		height=9, ratio=10, kind="hist", cbar=True, xlim=[0,200], ylim=[-90,90],
		joint_kws = dict(bins=(dist_bins, angle_bins)), marginal_ticks = True)
	g.ax_joint.set_xticks(dist_bins)
	g.ax_joint.set_yticks(angle_bins)
	g.ax_joint.set_xlabel('distance (ft.)')
	g.ax_joint.set_ylabel('angle (deg.)')
	g.fig.suptitle('Distance and Angle to No-Goals')
	g.fig.tight_layout()


def line_dist_goal_rate(df: pd.DataFrame):
	""" produces a histogram plot of goal rate w.r.t distance.
	:param df: dataframe with relevant features
	:return:
	"""
	#prepare data
	bins = np.arange(0, 201, 10)
	res = df.groupby(['event_type', pd.cut(df['shot_distance'], 
		bins)]).size().unstack().transpose()
	res = (res['GOAL'] / (res['GOAL'] + res['SHOT'])).round(3)
	res = res.dropna()
	x_points = np.vectorize(lambda x: (x.left + x.right)/2)(res.index)


	#plot
	plt.style.use('seaborn-deep')	
	plt.figure(figsize = (10, 5))
	plt.plot(x_points, np.asarray(res), 'o-')
	plt.title("Shot Distance to Goal Rate")
	plt.xlabel('distance (ft.)')
	plt.ylabel('goals / ( goals + shots)')
	plt.xticks(bins, bins)
	plt.show()


def line_angle_goal_rate(df: pd.DataFrame):
	""" produces a histogram plot of goal rate w.r.t angles.
	:param df: dataframe with relevant features
	:return:
	"""
	#prepare data
	bins = np.arange(-90, 91, 10)
	res = df.groupby(['event_type', pd.cut(df['shot_angle'], 
		bins)]).size().unstack().transpose()
	res = (res['GOAL'] / (res['GOAL'] + res['SHOT'])).round(3)
	res = res.dropna()
	x_points = np.vectorize(lambda x: (x.left + x.right)/2)(res.index)

	#plot
	plt.style.use('seaborn-deep')	
	plt.figure(figsize = (10, 5))
	plt.plot(x_points, np.asarray(res), 'o-')
	plt.title("Shot Angle to Goal Rate")
	plt.xlabel('angle (deg.)')
	plt.ylabel('goals / ( goals + shots)')
	plt.xticks(bins, bins)
	plt.show()

def hist_dist_goal_empty_net(df: pd.DataFrame):
	""" produces a histogram plot of goals scored when the net is empty ond not-empty w.r.t distance.
	:param df: dataframe with relevant features
	:return:
	"""
	#prepare data
	empty_net_goal = df.loc[(df['event_type'] == 'GOAL') & 
		(df['empty_net'] == True)]['shot_distance']
	non_empty_net_goal = df.loc[(df['event_type'] == 'GOAL') & 
		(df['empty_net'] == False)]['shot_distance']
	bins = np.arange(0, 201, 10)
	
	#plot
	plt.style.use('seaborn-deep')
	plt.figure(figsize = (10, 5))
	plt.hist([empty_net_goal, non_empty_net_goal], bins = bins, 
		rwidth = .95, label = ['empty net', 'non-empty net'])
	plt.yscale('log')
	plt.ylim((1, 10**5))
	plt.legend(loc='upper right')
	plt.title("Shot Distance to Empty/Non-Empty Net")
	plt.xlabel('distance (ft.)')
	plt.ylabel('Goal Count')
	plt.xticks(bins, bins)
	plt.show()

def hist_angle_goal_empty_net(df: pd.DataFrame):
	""" produces a histogram plot of goals scored when the net is empty ond not-empty w.r.t angles.
	:param df: dataframe with relevant features
	:return:
	"""
	#prepare data
	empty_net_goal = df.loc[(df['event_type'] == 'GOAL') & 
		(df['empty_net'] == True)]['shot_angle']
	non_empty_net_goal = df.loc[(df['event_type'] == 'GOAL') & 
		(df['empty_net'] == False)]['shot_angle']
	bins = np.arange(-90, 91, 10)
	
	#plot
	plt.style.use('seaborn-deep')
	plt.figure(figsize = (10, 5))
	plt.hist([empty_net_goal, non_empty_net_goal], bins = bins, 
		rwidth = .95, label = ['empty net', 'non-empty net'])
	plt.yscale('log')
	plt.ylim((1, 10**4))
	plt.legend(loc='upper right')
	plt.title("Shot Angle to Empty/Non-Empty Net")
	plt.xlabel('angle (deg.)')
	plt.ylabel('Goal Count')
	plt.xticks(bins, bins)
	plt.show()



