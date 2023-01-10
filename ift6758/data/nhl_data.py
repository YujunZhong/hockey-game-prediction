
import os
import sys

import errno
import pickle
import requests
import json

from tqdm import tqdm

import pandas as pd

BASE = 'https://statsapi.web.nhl.com/api/v1/'

# Team related APIs
TEAMS = 'teams'
TEAM_INFO = 'teams/{team_id}'
TEAM_ROASTER = f'{TEAM_INFO}/roaster'
TEAM_STATS = f'{TEAM_INFO}/stats'
POSITIONS = 'positions'

#Player related APIs
PLAYER_INFO = 'people/{player_id}'
PLAYER_STATS = f'{PLAYER_INFO}/stats'
STAT_TYPES = 'statTypes'

#Season related APIs
SEASONS = 'seasons'
SEASON = 'seasons/{season_id}'
CURRENT_SEASON = 'seasons/current'
STANDINGS = 'standings'
STANDING_TYPES = 'standingsTypes'

#Game related APIs
GAME_FEED = 'game/{game_id}/feed/live'
GAME_BOX_SCORE = 'game/{game_id}/boxscore'
GAME_LINE_SCORE = 'game/{game_id}/linescore'
GAME_TYPES = 'gameTypes'
GAME_STATUS = 'gameStatus'
PLAY_TYPES = 'playTypes'


class NHL_REST:
    '''
    NHL REST API client. 
    '''
    
    def __init__(self, url = BASE):
        '''
        :param url: base url (default is provided)
        '''
        self.url = url
    
    #--------  Player related requests  --------#

    def get_player_info(self, player_id):
        '''
        Get player info
        :param player_id: string
        '''
        req = self.__req(PLAYER_INFO.format(player_id = player_id))
        res = self.__get(req)
        return res

    def get_player_stats(self, player_id):
        req = self.__req(PLAYER_STATS.format(player_id = player_id))
        res = self.__get(req)
        return res
        
    #------- Player: Game related requests ------#


    #--------  Game related requests  --------#

    def get_game_feed(self, game_id):
        req = self.__req(GAME_FEED.format(game_id = game_id))
        res = self.__get(req)
        return res

    def get_play_types(self):
        req = self.__req(PLAY_TYPES)
        res = self.__get(req)
        return res

    #------- End: Game related requests ------# 
      
    def __get(self, req, params = dict()):
        res =  requests.get(req, params = params)
        if res.status_code == 200:
            return res.json()
        else:
            pass
            #print(f"There was an error during this http request: {req}\n status: {res.status_code}")
        
        return None
              
    def __req(self, api):
        return f'{self.url}{api}'




DEFAULT_SAVE_PATH = '/tmp/nhl'
LOAD_DATA_INFO = 'Loading {season} {game_type} season data...'
DOWNLOAD_DATA_INFO = 'Downloading {season} {game_type} season data...'
GAME_TYPE = { '02' :'regular', '03' : 'playoff' }
GAME_CNT = { '02' : 1271, '03' : 418}

class NHL:
    
    rest = NHL_REST()
    
    def __init__(self, start_year, end_year = None, path = DEFAULT_SAVE_PATH):
        if end_year == None:
            end_year = start_year
        self.target_years = list(range(start_year, end_year+1))
        if not os.path.isdir(path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        self.save_path = path
        self.data = []
        self.is_loaded = False
    
    def __add__(self, ):
        pass
    
    def to_DataFrame(self, event_types):
        df = None
        if len(self.data) > 0:
            df = Transformer.convert_to_dataframe(self.data, event_types)
        else:
            print("Data not loaded!")

        return df

    def to_DataFrame_optimized(self, event_types, game_types):
        dfs = []
        if len(self.target_years) > 0:
            for year in self.target_years:
                dfs.append(Transformer.convert_to_dataframe(
                    self.__get_season_data(year, game_types), event_types))
        else:
            print("No target years provided!")

        return pd.concat(dfs, ignore_index = True)
                
    def get_data(self, reload = False, game_types = list(GAME_TYPE.keys())):
        if reload or not self.is_loaded:
            del self.data[:]
            self.data = []
            for year in self.target_years:
                self.data.extend(self.__get_season_data(year, game_types))
            self.is_loaded = True

        return self.data  

    def __get_season_data(self, year, game_types):
        data = []
        season = f'{year}-{year+1}'
        
        sub_folder = os.path.join(self.save_path, f'year={year}')
        if not os.path.isdir(sub_folder):
            os.makedirs(sub_folder)
        
        for game_type in game_types:
            season_id = self.__gen_season_id(year, game_type)
            file_name = os.path.join(self.save_path, f'year={year}', f'{season_id}.pkl')
            game_type = GAME_TYPE[game_type]
            if os.path.exists(file_name):
                print(LOAD_DATA_INFO.format(season = season, game_type = game_type))
                
                data.extend(self.__load_data(file_name))
                print(f'Load success!')
            else:
                print(DOWNLOAD_DATA_INFO.format(season = season, game_type = game_type))
                data.extend(self.__download_data(season_id, file_name))
                print(f'Download success!')
        
        return data

    def __load_data(self, file_name):
        data = []
        with open(file_name, 'rb') as f:
            data = pickle.load(f)

        return data


    def __download_data(self, season_id, file_name):
        data = []
        game_ids = self.__gen_game_ids(season_id)
        for i in tqdm(range(len(game_ids))):
            game = self.rest.get_game_feed(game_ids[i])
            if game != None:
                data.append(game)
            
        with open(file_name, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        return data

    def __gen_season_id(self, year, game_type):
        return f'{year}{game_type}'
    
    def __gen_season_ids(self, year):
        return list(map(lambda s_type: gen_season_id(year, s_type), GAME_TYPE.keys()))

    def __gen_game_id(self, season_id, game_no): 
        return f'{season_id}{str(game_no).zfill(4)}'

    def __gen_game_ids(self, season_id):
        game_nos = self.__get_game_no(season_id[-2:])
        return list(map(lambda game_no: self.__gen_game_id(season_id, game_no), game_nos))
    
    def __get_game_no(self, game_type):
        game_nos = []
        if game_type == '02':
            game_nos = list(range(1, GAME_CNT[game_type]+1))
        elif game_type == '03':
            for r in range(1, 5):
                for m in range(1, 2**(4 - r) + 1):
                    game_nos.extend(list(map(lambda g: f'{r}{m}{g}', range(1, 8))))
        else:
            print(f"season type: {game_type} not supported!")
        return game_nos




GAME_DATA = 'gameData'
LIVE_DATA = 'liveData'
LINE_SCORE = 'linescore'
PLAYS = 'plays'
ALL_PLAYS = 'allPlays'
PENALTY_PLAYS = 'penaltyPlays'
SCORING_PLAYS = 'scoringPlays'
REQUIRED_GAME_STATE = '7'

DATE_TIME = 'dateTime'
PLAYERS = 'players'
RESULT = 'result'

EVENT_TYPE = 'eventTypeId'
ABOUT = 'about'
COORD = 'coordinates'

SECONDARY_TYPE = 'secondaryType'
EMPTY_NET = 'emptyNet'

PERIOD = 'period'
PERIODS = 'periods'
PERIOD_TIME = 'periodTime'


TEAM = 'team'
TEAMS = 'teams'
TEAM_CODE = 'triCode'
TEAM_NAME = 'name'
HOME = 'home'
AWAY = 'away'
RINK_SIDE = 'rinkSide'
X = 'x'
Y = 'y'


PROCESS_GAME_INFO = 'Processing Game: {game_id}'
    
class Transformer:
    """
    Transformer converts raw NHL data into a pandas DataFrame 
    with the set of columns considered relevant or 
    important.
    """

    column_name = ['season', 'game_id', 'game_type', 'date_time', 'period', 'period_time', 'event_type', 'team', 'team_code', 'rink_side',
                'coord_x', 'coord_y', 'shooter', 'goalie', 'shot_type', 'empty_net', 'strength', 'prev_event_type', 'prev_coord_x',
                'prev_coord_y', 'prev_period', 'prev_period_time', 'is_home', 'is_forward', 'is_shortHanded']
    
    @classmethod
    def penalty_time(cls, time, period, penalty_time):
        
        p_over = time.split(':')
        p_over[0] = str(int(time.split(':')[0])+penalty_time).zfill(2)
        if p_over[0] > '20':
            period +=1
            p_over[0] = str(int(p_over[0])-20).zfill(2)
            
        return ':'.join(p_over), period
    
    @classmethod
    def convert_to_dataframe(cls, data, event_types):
        
        data_frame = []
        print("Processing games...")
        for i in tqdm(range(len(data))): 
            game = data[i]
            entries = cls.__process_game(game, event_types)
            if entries:
                data_frame.extend(entries)

        return pd.DataFrame(data_frame, columns = cls.column_name)
    
    @classmethod
    def __process_game(cls, game, event_types):

        info = game[GAME_DATA]
        common = cls.__process_game_info(info)
        #print(PROCESS_GAME_INFO.format(game_id = common[1]))

        if info['status']['statusCode'] != REQUIRED_GAME_STATE or  LIVE_DATA not in game:
            return None

        plays, linescore = game[LIVE_DATA][PLAYS], game[LIVE_DATA][LINE_SCORE]

        periods = linescore[PERIODS]
        home = info[TEAMS][HOME][TEAM_CODE]
        away = info[TEAMS][AWAY][TEAM_CODE]
        rink_info = cls.__get_rink_info(periods, home, away)

        events = plays[ALL_PLAYS]
        penalties = plays[PENALTY_PLAYS]
        goals = plays[SCORING_PLAYS]

        t_data = []
        prev_event = None
        penalty_info = {info['teams']['home']['id']:[False, '00:00', 1, 0], info['teams']['away']['id']:[False, '00:00', 1, 0]}

        for event in events:
            event_type = event[RESULT][EVENT_TYPE]
            
            if event[RESULT]['event']=='Penalty':
                team_id = event['team']['id']
                other_team_id = list(penalty_info.keys())[list(penalty_info.keys()).index(team_id)-1]
                penalty_time = event[RESULT]['penaltyMinutes']
                penalty_info[team_id][3] = penalty_time
                penalty_info[team_id][0] = True
                penalty_info[team_id][1], penalty_info[team_id][2] = cls.penalty_time(event['about']['periodTime'], event['about']['period'], penalty_time)
            else:
                if event['about']['periodTime'] > list(penalty_info.values())[0][1] and list(penalty_info.values())[0][2] == event['about']['period']:
                    list(penalty_info.values())[0][0] = False
                if event['about']['periodTime'] > list(penalty_info.values())[1][1] and list(penalty_info.values())[1][2] == event['about']['period']:
                    list(penalty_info.values())[1][0] = False
            
            if event['about']['periodType']!='SHOOTOUT':
                if event_type in event_types:
                    is_shortHanded = 0
                    team_id = event['team']['id']
                    other_team_id = list(penalty_info.keys())[list(penalty_info.keys()).index(team_id)-1]
                    if (penalty_info[team_id][0] == False) and (penalty_info[other_team_id][0] == True):
                        is_shortHanded = 1
                    if event[RESULT]['event']=='Goal':
                        if list(penalty_info.values())[0][3]==2:
                            list(penalty_info.values())[0][0] = False
                        if list(penalty_info.values())[1][3]==2:
                            list(penalty_info.values())[1][0] = False
                    row = cls.__process_event(game, event, prev_event, rink_info, is_shortHanded)
                    if row:
                        t_data.append(common + row)
                prev_event = event
        return t_data
    
    @classmethod
    def __process_game_info(cls, info):
        row = []
        row.append(f"{info['game']['season'][:4]}-{info['game']['season'][4:]}")
        row.append(info['game']['pk'])
        row.append(info['game']['type'])
        #row.append(info['datetime']['dataTime'])
        return row

    @classmethod
    def __process_event(cls, game, event, prev_event, rink_info, is_shortHanded):
        
        result = event[RESULT]
        about = event[ABOUT]
        coord = event[COORD]
        event_type = result[EVENT_TYPE]

        if X not in coord or Y not in coord or TEAM not in event or PLAYERS not in event :
            return None

        team = event[TEAM]
        team_code = team[TEAM_CODE]
        players = event[PLAYERS]
        period = about[PERIOD]
        rink_side = rink_info[team_code][(period - 1) % 2]

        row = []
        row.append(about[DATE_TIME])
        row.append(period)
        row.append(about[PERIOD_TIME])
        row.append(event_type)
        row.append(team[TEAM_NAME])
        row.append(team_code)
        row.append(rink_side)
        row.append(coord[X])
        row.append(coord[Y])
        row.append(players[0]['player']['fullName'])
        row.append(players[1]['player']['fullName'] if len(players) > 1 else None)
        # shot type
        row.append(result[SECONDARY_TYPE] if SECONDARY_TYPE in result else None)
        row.append(result[EMPTY_NET] if EMPTY_NET in result else None)
        row.append(result['strength']['name'] if event_type == 'GOAL' else None)

        # And previous event info
        prev_event_type = None
        prev_coord_x = None
        prev_coord_y = None
        prev_period = None
        prev_period_time = None
        if prev_event != None:
            prev_event_type = prev_event[RESULT][EVENT_TYPE]
            prev_period = prev_event[ABOUT][PERIOD]
            prev_period_time = prev_event[ABOUT][PERIOD_TIME]
            prev_coord = prev_event[COORD]
            if X in prev_coord and Y in prev_coord:
                prev_coord_x = prev_coord[X]
                prev_coord_y = prev_coord[Y]

        row.append(prev_event_type)
        row.append(prev_coord_x)
        row.append(prev_coord_y)
        row.append(prev_period)
        row.append(prev_period_time)
        
        row.append(1 if game['gameData']['teams']['home']['name'] == team['name'] else 0)
        player_id = players[0]['player']['id']
        key_id = "ID{}".format(player_id)
        row.append(1 if game['gameData']['players'][key_id]["primaryPosition"]["type"] == 'Forward' else 0)
        row.append(is_shortHanded)
        
        return row

    @classmethod
    def __get_rink_info(cls, periods, home, away):
        default_side = ['left', 'right']
        period = periods[0]
        home_side = period[HOME][RINK_SIDE] if RINK_SIDE in period[HOME] else default_side[0]
        away_side = period[AWAY][RINK_SIDE] if RINK_SIDE in period[AWAY] else default_side[1]
        rink_info = {home: [home_side, away_side], away: [away_side, home_side]}

        return rink_info

    def __get_team_info(cls, team):
        return team[TEAM_CODE]



        
    