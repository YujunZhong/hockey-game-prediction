import os
import sys
import requests
import pandas as pd
import pickle


class NHL:
    """download and process NHL data."""
    API_URL = "https://statsapi.web.nhl.com/api/v1/game/"
    game_number = 1271

    def __init__(self, target_year, save_path):
        """
        :param target_year: target year
        :param save_path: the folder where the data are saved.
        :return:
        """
        self.target_year = target_year
        self.save_path = save_path

    def __add__(self, dataset, newdataset):
        return dataset.update(newdataset)

    def get_data(self):
        """ download or locally load NHL play-by-play data for both the regular season and playoffs.
        :return:
        """
        dataset = []
        game_type = {"02": "regular_season", "03": "playoffs"}
        for season in game_type.keys():
            file_name = os.path.join(self.save_path, f"{self.target_year}_{game_type[season]}.pkl")

            if os.path.exists(file_name):
                print(f"Loading {int(self.target_year)}-{int(self.target_year)+1} {game_type[season]} data")
                with open(file_name, 'rb') as f:
                    dataset = pickle.load(f)
            else:
                print(f"Downloading {int(self.target_year)}-{int(self.target_year)+1} {game_type[season]} data")
                # get the game ids we need
                game_id = self.target_year + season
                for i in range(NHL.game_number):
                    response = requests.get(url= NHL.API_URL + game_id + str(i).zfill(4)+ '/feed/live')
                    if response.status_code == 200:
                        data = response.json()
                        dataset.append(data)
                    else:
                        continue

                with open(file_name, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

        return dataset
    
    def tidy_data(self, dataset, events_type):
        """ processing the raw event data from every game into dataframes
        :param dataset: raw data
        :param events_type: what types of events we want to format
        :return:
        """
        games_data = []

        for data in dataset:
            if 'liveData' not in data:
                continue

            games = data['liveData']['plays']['allPlays']
            for game in games:
                for event in events_type:
                    data_shot_goal = []
                    if game['result']['event'] in [event]:
                        # gameTime
                        data_shot_goal.append(data['gameData']['datetime']['dateTime'])
                        # gameId
                        data_shot_goal.append(data['gamePk'])
                        # team name
                        data_shot_goal.append(game['team']['name'])
                        # data_shot_goal.append(data['gameData']['teams']['away']['name'])
                        # shot or goal
                        data_shot_goal.append(game['result']['event'])
                        # coordinates
                        if 'x' in game['coordinates'] and 'y' in game['coordinates']:
                            data_shot_goal.append(game['coordinates']['x'])
                            data_shot_goal.append(game['coordinates']['y'])
                        else:
                            continue
                        # shooter name
                        data_shot_goal.append(game['players'][0]['player']['fullName'])
                        # Goalie name
                        if len(game['players']) > 1:
                            data_shot_goal.append(game['players'][1]['player']['fullName'])
                        else:
                            data_shot_goal.append("None")  
                        # shot type
                        if 'secondaryType' in game['result']:
                            data_shot_goal.append(game['result']['secondaryType'])
                        else:
                            data_shot_goal.append("None")
                        if 'emptyNet' in game['result']:
                            data_shot_goal.append(game['result']['emptyNet'])
                        else:
                            data_shot_goal.append("None")
                            
                        if event == 'Goal':
                            data_shot_goal.append(game['result']['strength']['name'])
                        else:
                            data_shot_goal.append("None")

                        games_data.append(data_shot_goal)

        column_name = ['gameTime', 'gameId', 'team', 'eventType', 'coordinates_x', 'coordinates_y', 'shooter', 'goalie', 'shotType', 'emptyNet', 'strength']
        df = pd.DataFrame(games_data, columns = column_name)

        return df


def get_data(start_season, end_season, data_path):
    """ get the raw json format data for the seasons we want
    :param start_season: the first year of the data we want
    :param end_season: the last year of the data we want
    :param data_path: the folder where the data are saved
    :return:
    """
    for tar in range(start_season, end_season+1):
        if not os.path.isdir(data_path):
            os.makedirs(data_path)

        nhl_class = NHL(str(tar), data_path)
        raw_data = nhl_class.get_data()
    return raw_data


def tidy_data(start_season, end_season, data_path, events_type):
    """ processing the raw event data from every game into dataframes
    :param start_season: the first year of the data we want
    :param end_season: the last year of the data we want
    :param data_path: the folder where the data are saved
    :param events_type: events of the type we want to include
    :return:
    """
    for yr in range(start_season, end_season+1):
        csv_path = os.path.join(data_path, f'{yr}.csv')
        if os.path.exists(csv_path):
            continue
        nhl_class = NHL(str(yr), data_path)
        dataset = nhl_class.get_data()
        cleaned_data = nhl_class.tidy_data(dataset, events_type)
        cleaned_data.to_csv(os.path.join(data_path, f'{yr}.csv'))


if __name__ == "__main__":
    start_season = 2016
    end_season = 2020
    data_path = sys.argv[1]

    # download or load raw data
    raw_data = get_data(start_season, end_season, data_path)

    # download or load raw data, and transform into dataframe
    events_type = ['Shot','Goal']
    tidy_data(start_season, end_season, data_path, events_type)
