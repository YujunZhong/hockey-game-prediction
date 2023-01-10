import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

import os
import cv2
from PIL import Image

from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter

import plotly.graph_objects as go


def func1(d):
    """ Calculate the average shot rate per hours for all teams (used in binned_statistic_2d)
    :param d:
    :return:
    """
    global total_num_games
    return (sum(d) / 2) / total_num_games


def func2(d):
    """ Calculate the average shot rate per hours for one team (used in binned_statistic_2d)
    :param d:
    :return:
    """
    global team_num_games
    return sum(d) / team_num_games


def convert_xy_to_half_from_net(data):
    """ Convert coordinates 
    :param data: the raw dataframe
    :return: dataframe with new coordinates x and y
    """
    drop_list = []
    for index, row in data.iterrows():
        if abs(row['coordinates_x']) > 89.0:
            drop_list.append(index)
            continue

        if row['coordinates_x'] < 0.:
            data.at[index,'coordinates_y'] = -1 * row['coordinates_y']
        data.at[index,'coordinates_x'] = 89.0 - abs(row['coordinates_x'])
    data = data.drop(drop_list)
    
    return data


def get_total_shot_rate(data, x_bin, y_bin):
    """ Compute shot rates for all teams
    :param data: the input dataframe
    :param x_bin: x bins
    :param y_bin: y bins
    :return: shot rate and bin edges
    """
    x_shots = data['coordinates_x'].tolist()
    y_shots = data['coordinates_y'].tolist()

    z = [1] * len(x_shots)
    shot_rate, x_edge, y_edge, _ = binned_statistic_2d(x_shots, y_shots, values = z, statistic =func1, bins=[x_bin, y_bin])

    return shot_rate, x_edge, y_edge


def get_team_shot_rate(data, x_bin, y_bin):
    """ Compute shot rates for one specific team
    :param data: the input dataframe
    :param x_bin: x bins
    :param y_bin: y bins
    :return: shot rate and bin edges
    """
    x_team_shot = data['coordinates_x'].tolist()
    y_team_shot = data['coordinates_y'].tolist()

    z = [1] * len(x_team_shot)
    team_shot_rate, x_edge, y_edge, _ = binned_statistic_2d(x_team_shot, y_team_shot, values = z, statistic = func2, bins=[x_bin, y_bin])

    return team_shot_rate, x_edge, y_edge


def shot_maps_plt(data, team_name, shot_rate, x_bin, y_bin, sigma, half_rink):
    """ create shot maps with matplotlib.
    :param data: the input dataframe
    :param team_name: which team we want to show
    :param shot_rate: the shot rates of all teams
    :param x_bin: x bins
    :param y_bin: y bins
    :param sigma: standard deviation for Gaussian kernel
    :param half_rink: the half-rink image path 
    :return:
    """
    global team_num_games

    # Group data by team
    team_data = data.groupby('team')
    df = team_data.get_group(team_name)
    team_num_games = df['gameId'].nunique()

    team_shot_rate, x_edge, y_edge = get_team_shot_rate(df, x_bin, y_bin)

    diff_shot_rate = team_shot_rate - shot_rate

    # Smooth the data
    diff_shot_rate_norm = gaussian_filter(diff_shot_rate,sigma = sigma)
    
    x_centers = (x_edge[:-1] + x_edge[1:]) / 2
    y_centers = (y_edge[:-1] + y_edge[1:]) / 2
        
    data_min= diff_shot_rate_norm.min()
    data_max= diff_shot_rate_norm.max()

    max_total = 0
    if abs(data_min) > data_max:
        max_total = abs(data_min)
    elif data_max > abs(data_min):
        max_total = abs(data_max)

    diff_shot_rate_norm = diff_shot_rate_norm / max_total
    
    data_min = diff_shot_rate_norm.min()
    data_max = diff_shot_rate_norm.max()
    
    if abs(data_min) > data_max:
        data_max = data_min * -1
    elif data_max > abs(data_min):
        data_min = data_max * -1

    diff_shot_rate_norm = diff_shot_rate_norm.T

    I = Image.open(half_rink)

    # Without rink
    fig, ax = plt.subplots(figsize=(10,12), facecolor='w', edgecolor='k')
    plt.contourf(x_centers,y_centers,diff_shot_rate_norm,alpha = 1.0, cmap='bwr', 
            levels = np.linspace(data_min,data_max,20),
            vmin=data_min,
            vmax=data_max,
                )
    ax.invert_xaxis()
    plt.colorbar(orientation = 'horizontal', pad = 0.05)
    plt.title('Excess shots rate',fontdict={'fontsize': 15})
    plt.show()

    # Wit rink
    fig, ax = plt.subplots(figsize=(8.5,10), facecolor='w', edgecolor='k')
    contour_image = ax.contourf(x_centers, y_centers, diff_shot_rate_norm,alpha = 0.7, cmap='bwr', 
            levels = np.linspace(data_min,data_max,20),
            vmin=data_min,
            vmax=data_max,
                )

    ax.invert_xaxis()
    rink_image = ax.imshow(I, extent = (89, -11, -42.5, 42.5), alpha=1., origin='lower')

    fig.colorbar(contour_image, orientation = 'horizontal', pad = 0.05)
    plt.title('Excess shots rate',fontdict={'fontsize': 15})

    plt.show()
    

def create_dropdown(team_name, year, teams):
    return dict(label = team_name, method='update', args=[{'visible': teams.columns.isin([team_name]),
        'title': team_name,
        'showlegend': True},
        {'title': {'text': f'{year}-{year+1} season shot rates', 'x': 0.5}}])


def shot_maps_plotly(data, year, shot_rate, x_bin, y_bin, sigma, output_path, half_rink):
    """ create shot maps with plotly.
    :param data: the input dataframe
    :param year: the target year we want to visualize
    :param shot_rate: the shot rates of all teams
    :param x_bin: x bins
    :param y_bin: y bins
    :param sigma: standard deviation for Gaussian kernel
    :param output_path: exported html file path 
    :param half_rink: the half-rink image path 
    :return:
    """
    global team_num_games

    fig = go.Figure(layout=go.Layout(height=620, width=700))
    teams = pd.DataFrame(columns=data['team'].unique())
    
    # Group data by team
    team_data = data.groupby('team')

    for team_name in data['team'].unique():
        visible = False
        if team_name == 'Winnipeg Jets':
            visible = True

        df = team_data.get_group(team_name)
        team_num_games = df['gameId'].nunique()

        team_shot_rate, x_edge, y_edge = get_team_shot_rate(df, x_bin, y_bin)
        diff_shot_rate = team_shot_rate - shot_rate

        # Smooth the data
        diff_shot_rate_norm = gaussian_filter(diff_shot_rate,sigma = sigma)

        x_centers = (x_edge[:-1] + x_edge[1:]) / 2
        y_centers = (y_edge[:-1] + y_edge[1:]) / 2
        x_centers[-1] = 89.0
            
        data_min= diff_shot_rate_norm.min()
        data_max= diff_shot_rate_norm.max()

        max_total = 0
        if abs(data_min) > data_max:
            max_total = abs(data_min)
        elif data_max > abs(data_min):
            max_total = abs(data_max)

        diff_shot_rate_norm = diff_shot_rate_norm / max_total
        
        data_min = diff_shot_rate_norm.min()
        data_max = diff_shot_rate_norm.max()
        
        if abs(data_min) > data_max:
            data_max = data_min * -1
        elif data_max > abs(data_min):
            data_min = data_max * -1

        diff_shot_rate_norm = diff_shot_rate_norm.T

        fig.add_trace(go.Contour(
            z = diff_shot_rate_norm,
            y = y_centers,
            x = x_centers,
            colorscale=[[0, 'blue'], [0.5, 'white'], [1, 'red']],
            zmin=-1.0,
            zmax=1.0,
            opacity=0.5,
            visible=visible
        ))
    
    fig.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active = 0,
            buttons = list(teams.columns.map(lambda team_name: create_dropdown(team_name, year, teams))),
            x=0.1,
            xanchor="left",
            y=1.1,
            yanchor="top"
        )]
    )

    fig.add_layout_image(
        dict(
            source=Image.open(half_rink),
            xref="x",
            yref="y",
            x=89.0,
            y=42.5,
            sizex=100,
            sizey=85,
            sizing="stretch",
            layer='below',
            opacity=1.0,
        )
    )

    # Add annotation
    fig.update_layout(
        annotations=[
            dict(text="Team:", showarrow=False,
            x=95, y=1.085, yref="paper", align="left")
        ]
    )

    fig['layout']['xaxis']['autorange'] = "reversed"

    fig.show()

    fig.write_html(output_path)


def crop_half_rink(rink_file, half_rink):
    """ crop NHL rink image into half rink.
    :param rink_file: the full rink image path
    :param half_rink: the output half-rink image path 
    :return:
    """
    if not os.path.exists(half_rink):
        img = cv2.imread(rink_file)

        half = img.shape[1] // 2
        right_part = img[:, half:]  
        cv2.imwrite(half_rink, right_part)


def advanced_visualizations(year, half_rink):
    """ create shot maps with plotly or matplotlib.
    :param year: the target year we want to visualize
    :param half_rink: the half-rink image path 
    :return:
    """
    # Insert the Data (panda data frame)
    data = pd.read_csv(f'./ift6758/data/NHL/{year}.csv')
    output_path = f"./shot_map_{year}.html"

    data = convert_xy_to_half_from_net(data)

    global total_num_games
    total_num_games = data['gameId'].nunique()

    # Define bin size (#of bins in x and y coordinates) -> (you can change it)
    binsize = (4,5)

    # Define sigma for gaussian_filter -> (you can change it)
    sigma = 1.

    x_range = (0, 100)
    y_range = (-42.5,42.5)
    x_bin = np.arange(x_range[0] - binsize[0] / 2, x_range[1] + binsize[0], binsize[0])
    y_bin = np.arange(y_range[0] - binsize[1] / 2, y_range[1] + binsize[1], binsize[1])

    shot_rate, _, _ = get_total_shot_rate(data, x_bin, y_bin)

    shot_maps_plotly(data, year, shot_rate, x_bin, y_bin, sigma, output_path, half_rink)
    # shot_maps_plt(data, 'Winnipeg Jets', shot_rate, x_bin, y_bin, sigma, half_rink)


if __name__== "__main__":
    rink_file = "./figures/nhl_rink.png"
    half_rink = "./figures/right.jpg"
    crop_half_rink(rink_file, half_rink)

    year = 2017
    advanced_visualizations(year, half_rink)