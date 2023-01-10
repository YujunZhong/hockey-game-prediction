import pandas as pd
import numpy as np


def clean_data(inp_df: pd.DataFrame):
	df = inp_df.copy()
	popular_shot_type = df['shot_type'].value_counts().index[0]
	df.loc[df['shot_type'].isna(), 'shot_type'] = popular_shot_type
	df.loc[df['empty_net'].isna(), 'empty_net'] = False
	df.loc[df['strength'].isna(), 'strength'] = 'Even'

	return df

