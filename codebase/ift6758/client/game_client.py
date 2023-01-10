import os
import sys
import requests
import datetime
from datetime import datetime as dt

from ift6758.data import NHL
from ift6758.data import clean_data
from ift6758.feature import extract_features, transform_features


class GameClient:
    
    def __init__(self):
        pass

    def get_game_status(self, gameId):
        game_id = 'https://statsapi.web.nhl.com/api/v1/game/' + str(gameId) + '/feed/live'
        game_data = requests.get(game_id).json() 
        gameState = game_data['gameData']['status']['abstractGameState']

        if gameState == 'Final': 
            result = {'Home_Team' : game_data['gameData']['teams']['home']['name'],
                 'Away_Team' : game_data['gameData']['teams']['away']['name'],
                 'Game_ID' : game_data['gamePk']}
            return ('FINAL', result)

        elif gameState == 'Preview':
            start_time = game_data['gameData']['datetime']['dateTime']
            time = dt.strptime(start_time, '%Y-%m-%dT%H:%M:%SZ')
            hour = int(time.strftime('%I')) + 7 
            second = time.strftime(':%M')
            if hour >= 13:
                return ('Preview', f"{str(hour - 12)} {second} PM")
            else:
                return ('Preview', f"{hour} {second} PM")

        else:
            time_left =  game_data['liveData']['plays']['allPlays'][len(game_data['liveData']['plays']['allPlays'])-1]['about']['periodTimeRemaining']
            game_period = game_data['liveData']['plays']['allPlays'][len(game_data['liveData']['plays']['allPlays'])-1]['about']['ordinalNum']
            result = {'Home_Team' : game_data['gameData']['teams']['home']['name'],
                 'Away_Team' : game_data['gameData']['teams']['away']['name'],
                 'Game_ID' : game_data['gamePk'],
                 'Game_Status' : (game_period, time_left)}
            return ('live', result)
        
    def get_today_games(self):
        date = datetime.date.today().strftime("%Y-%m-%d")
        game_url = 'https://statsapi.web.nhl.com/api/v1/schedule?startDate='+ date +'&endDate=' + date
        live_games = requests.get(game_url).json()
        return live_games

    def get_game(self, game_info):
        status, game = self.get_game_status(game_info['gamePk'])
        result = {'Home_Team' : game_info['teams']['home']['team']['name'],
                 'Away_Team' : game_info['teams']['away']['team']['name'],
                 'Game_ID' : game_info['gamePk'],
                 'Game_Status' : (status, game)}
        return result
    
    def get_game_df(self, gameId):
        nhl = NHL(2022, path = None)
        event_types = ['SHOT', 'GOAL']
        df = nhl.game_to_DataFrame(gameId, event_types)
        if not df.empty:
            # Clean data
            clean_df = clean_data(df)

            # Feature extraction
            game_df = extract_features(clean_df)
            game_df = transform_features(game_df)
        else:
            game_df = df
        
        return game_df

