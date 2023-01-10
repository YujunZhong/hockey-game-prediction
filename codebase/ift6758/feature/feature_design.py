import numpy as np
import pandas as pd


def extract_features(input_df: pd.DataFrame):
	""" Extract and compute features from raw input dataframe
	:param df: raw input dataframe
	:return: DataFrame with important features
	"""
	df = get_goal_side(input_df)
	df['game_seconds'] = compute_game_seconds(df)
	df['period_seconds'] = compute_period_seconds(df['period_time'])
	df['shot_distance'] = compute_shot_distance(df, 'coord_x', 'coord_y')
	df['shot_angle'] = compute_shot_angle(df, 'coord_x', 'coord_y')
	df['distance_from_prev_event'] = compute_dist_from_prev_event(df)
	df['time_from_prev_event'] = compute_time_from_prev_event(df)
	df['rebound'] = compute_rebound(df)
	df['change_in_angle'] = compute_angle_change(df)
	df['speed'] = compute_speed(df)
	columns_order = ['date_time', 'season', 'game_id', 'game_type', 'game_seconds',
		'period', 'period_time', 'period_seconds','event_type', 'coord_x', 'coord_y',
		'prev_period_time', 'prev_event_type', 'prev_coord_x', 'prev_coord_y',
		'team', 'team_code', 'goal_side', 'shooter', 'goalie', 'shot_distance', 'shot_angle',
		'distance_from_prev_event', 'time_from_prev_event', 'rebound', 'change_in_angle',
		'speed', 'shot_type', 'empty_net', 'strength', 'is_home', 'is_forward', 'is_shortHanded']
	return df[columns_order]



#--------       Shot distances and angles ------#
goal_coord = { 'left': (-89, 0), 'right' : (89, 0) }

def get_goal_side(df: pd.DataFrame):
	""" Gets goal side based on events of a game.
	:param df: input dataframe
	:return:  dataframe with goal_side information .
	"""
	tmp = df.groupby(['game_id', 'period', 'team_code'])[['coord_x']].sum()
	tmp['goal_side'] = np.where(tmp['coord_x'] > 0, 'right', 'left')
	return df.join(tmp['goal_side'], ['game_id', 'period', 'team_code'])


def get_goal_coords(goal_side):
	""" Returns the goal coordinates based on goal side
	:param goal side: str
	:return: goal 2d coordinates (tuple)
	"""
	g_x, g_y = np.vectorize(lambda side: goal_coord[side])(goal_side)
	return np.concatenate((g_x[:, None], g_y[:, None]), axis=1)

def compute_shot_distance(df: pd.DataFrame, coord_x, coord_y):
	""" Compute shot distance from the goal post.
	:param df: input dataframe
	:param coord_x: column name which contains the x-coordinate of the event
	:param coord_y: column name which contains the y-coordinate of the event
	:return:  goal shot distance (Series).
	"""
	g_coords = get_goal_coords(df['goal_side'])
	pos = df[[coord_x, coord_y]].to_numpy()
	return pd.Series(np.linalg.norm(pos - g_coords, axis=1)).round(2)

def compute_shot_angle(df: pd.DataFrame, coord_x, coord_y):
	""" Compute shot angle from the goal post.
	:param df: input dataframe
	:param coord_x: column name which contains the x-coordinate of the event
	:param coord_y: column name which contains the y-coordinate of the event
	:return:  goal shot angle (Series).
	"""
	#hyp = compute_shot_distance(df, coord_x, coord_y)
	opp = df[coord_y]
	g_coord_x = get_goal_coords(df['goal_side'])[:,0]
	adj = g_coord_x - df[coord_x]
	return pd.Series(np.arctan(opp / adj) * 180/np.pi).round(2)


#-----------     Period seconds and Games Seconds -------#
def period_seconds(t: str):
	""" Compute period seconds of an event.
	:param t: time(str)
	:return:  period seconds (int).
	"""
	m, sec = tuple(t.strip().split(':'))
	return int(m) * 60 + int(sec)

def compute_period_seconds(s: pd.Series):
	""" Compute period seconds of an event.
	:param s: period time (Series)
	:return:  period seconds (Series).
	"""
	return pd.Series(np.vectorize(period_seconds)(s))

def compute_game_seconds(df: pd.DataFrame):
	""" Compute game seconds of an event.
	:param df: input dataframe
	:return:  game seconds (Series).
	"""
	period_seconds = compute_period_seconds(df['period_time'])

	PERIOD_DUR = 1200 
	OVER_TIME_DUR = np.where(df['game_type'] == 'R', 300, PERIOD_DUR)
	period = np.minimum(3, df['period'] - 1)
	over_time = np.maximum(0, df['period'] - 4)

	return PERIOD_DUR * period + OVER_TIME_DUR * over_time + period_seconds



#--------   Previous event - Distnace, time, rebound, angle change and speed ---- #
def compute_dist_from_prev_event(df: pd.DataFrame):
	""" Compute distance between 2 consecutive events.
	:param df: input dataframe
	:return: distance (Series).
	"""
	prev = df[['prev_coord_x', 'prev_coord_y']].to_numpy()
	curr = df[['coord_x', 'coord_y']].to_numpy()
	return pd.Series(np.linalg.norm(curr - prev, axis=1)).round(2)

def compute_time_from_prev_event(df: pd.DataFrame):
	""" Compute time between 2 consecutive events.
	:param df: input dataframe
	:return: time gap (Series).
	"""
	curr_period_seconds = df['period_seconds']
	prev_period_seconds = compute_period_seconds(df['prev_period_time'])
	return curr_period_seconds - prev_period_seconds

def compute_rebound(df: pd.DataFrame):
	""" Compute whether the shot was a rebound.
	:param df: input dataframe
	:return: Series[bool] indicating rebound.
	"""
	return df['prev_event_type'] == 'SHOT'

def compute_angle_change(df: pd.DataFrame):
	""" Compute the change in angle between 2 events.
	:param df: input dataframe
	:return: Series containing angle change.
	"""
	curr_shot_angle = df['shot_angle']
	prev_shot_angle = compute_shot_angle(df, 'prev_coord_x', 'prev_coord_y')
	return pd.Series(np.where( df['rebound'], (curr_shot_angle - prev_shot_angle).abs(), 0))

def compute_speed(df: pd.DataFrame):
	""" Compute the speed.
	:param df: input dataframe
	:return: Series containing speed values
	"""
	return (df['distance_from_prev_event'] / df['time_from_prev_event']).round(2)



#---------- Feature Transform for model training ------------------#

event_types = {'SHOT': 0, 'GOAL': 1, 'BLOCKED_SHOT': 2, 
            'CHALLENGE': 3, 'FACEOFF': 4, 'GIVEAWAY': 5, 
            'HIT': 6, 'MISSED_SHOT': 7, 'PENALTY': 8, 
            'PERIOD_END': 9, 'PERIOD_READY': 10, 
            'PERIOD_START': 11, 'STOP': 12, 
            'TAKEAWAY': 13, 'PERIOD_OFFICIAL': 14, 
            'SHOOTOUT_COMPLETE': 15, 'GAME_OFFICIAL': 16, 'GAME_END' : 17}

shot_types = {'Backhand': 1, 'Deflected': 2, 
            'Slap Shot': 3, 'Snap Shot': 4, 
            'Tip-In': 5, 'Wrap-around': 6, 
            'Wrist Shot': 7, 'Batted': 8, 'Poke': 9}

feature_list = ['game_seconds', 'period', 'coord_x', 'coord_y', 
            'shot_distance', 'shot_angle', 'shot_type', 
            'prev_event_type', 'prev_coord_x', 'prev_coord_y', 
            'time_from_prev_event', 'distance_from_prev_event', 
            'rebound', 'change_in_angle', 'speed', 'empty_net']

target = 'event_type'

def transform_features(inp_df: pd.DataFrame):
	""" Transform the features to numerical form to be used for training by
	 the learning algorithms
	:param df:  input dataframe
	:return: dataframe compatible for training
	"""
	df = inp_df.copy()
	# change format into our need 
	df['shot_angle'] = df['shot_angle'].abs()

	#set(df.prev_event_type)
	df['event_type'].replace(event_types, inplace=True)
	df['prev_event_type'].replace(event_types, inplace=True)
	df['prev_event_type'] = pd.to_numeric(df['prev_event_type'])

	df['shot_type'].replace(shot_types, inplace=True)

	bool_to_int = {False: 0, True: 1}
	df['rebound'].replace(bool_to_int, inplace=True)
	df['speed'].replace({np.inf: 0}, inplace=True)
	df['empty_net'].replace(bool_to_int, inplace=True)
	df['empty_net'] = pd.to_numeric(df['empty_net'])

	for f in feature_list:
		df[f] = df[f].fillna(0)

	return df



#---------------   Features Selection -----------------#


def sep_feature_target(df: pd.DataFrame, features):
	""" Select features from dataframe and separate features and target variable.
	:param df: input dataframe
	:param features: list of features to select
 	:return:  features, target (Tuple).
	"""
	return df[features].to_numpy(), df['event_type'].to_numpy()
