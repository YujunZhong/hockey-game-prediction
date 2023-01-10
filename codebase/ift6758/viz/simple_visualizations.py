import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compare_shot_types(df):
    """ Compare shots and goals for different shot types.
    :param df: input dataframe (raw)
    :return:
    """
    new_df = df.groupby(['shotType', 'eventType']).size().to_frame('counts').reset_index()
    new_df = new_df.pivot(index='shotType', columns=['eventType'], values=['counts'])

    new_df = new_df.drop(['None'])
    new_df['total'] = new_df.sum(axis=1)
    new_df['success rate'] = new_df.apply(lambda row: row['counts']['Goal'] / row['total'], axis=1)
    new_df['success rate'] = new_df['success rate'].astype(float).map("{:.2%}".format)

    ax = new_df['total'].plot(kind='bar', color='dodgerblue', label='Total shots', title='Compare different shot types')
    ax = new_df['counts']['Goal'].plot(kind='bar', color='red', label='Goals')
    plt.legend()
    plt.xticks(rotation=15)
    plt.xlabel('Shot type')
    plt.ylabel('Counts')

    ax.bar_label(ax.containers[1], labels=new_df['success rate'], color='black', label_type='edge', fontsize='x-small')
    plt.show()


def goal_and_shot_distances(df):
    """ visualize the relationship between the distance a shot was taken and the chance it was a goal.
    :param df: input dataframe (raw)
    :return:
    """
    df['distances'] = df.apply(lambda row: ((89 - np.abs(row['coordinates_x'])) ** 2 + row['coordinates_y'] ** 2) ** (1/2), axis=1)
    bins = list(range(1, 102, 10))
    df['bins'] = pd.cut(df['distances'], bins)
    groups = df.groupby(df['bins'])['eventType', 'distances']

    dist_list = []
    for key, _ in groups:
        group = groups.get_group(key)
        goal_perc = len(group[group['eventType'] == 'Goal']) / (len(group['eventType']))
        dist_list.append([key, goal_perc])

    new_df = pd.DataFrame(dist_list, columns=["Range of distance", "Goal percentage"])
    new_df.plot.line(x='Range of distance', y='Goal percentage', **{'marker': 'o'})
    plt.title('Shot distance and goal percentage')
    plt.xlabel('Distance')
    plt.ylabel('Goal percentage')

    plt.show()


def goal_and_shot_distances_types(df):
    """ visualize the relationship between the distance a shot was taken and the chance it was a goal, displayed separately under different shooting types.
    :param df: input dataframe (raw)
    :return:
    """
    df['distances'] = df.apply(lambda row: ((89 - np.abs(row['coordinates_x'])) ** 2 + row['coordinates_y'] ** 2) ** (1/2), axis=1)
    bins = list(range(1, 102, 10))
    dist_cut = pd.cut(df['distances'], bins)
    groups = df.groupby(dist_cut)['eventType', 'shotType', 'distances']

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 5)

    shot_types = ['Backhand', 'Deflected', 'Slap Shot', 'Snap Shot', 'Tip-In', 'Wrap-around', 'Wrist Shot']
    for ty in shot_types:
        dist_list = []
        for key, _ in groups:
            group = groups.get_group(key)
            group = group[group['shotType'] == ty]
            if len(group) >= 30:
                goal_perc = len(group[group['eventType'] == 'Goal']) / (len(group))
                dist_list.append([key, goal_perc])
            else:
                dist_list.append([key, None])

        new_df = pd.DataFrame(dist_list, columns=["Range of distance", "Goal percentage"])
        ax = new_df.plot(ax=ax, kind='line', x='Range of distance', y='Goal percentage', label=ty, **{'marker': 'o'})
        plt.title('Goal percentage by different shot types and distances (filter out distances with less than 30 occurrences)')
        plt.xlabel('Distance')
        plt.ylabel('Goal percentage')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    data_path = sys.argv[1]  # data loading path, for example "../data/NHL"
    target_season = 2019
    df = pd.read_csv(os.path.join(data_path, f"{target_season}.csv"))

    # compare_shot_types(df)
    goal_and_shot_distances(df)
    # goal_and_shot_distances_types(df)
