import requests
import json

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
            print(f"There was an error during this http request: {req}\n status: {res.status_code}")
        
        return None
              
    def __req(self, api):
        return f'{self.url}{api}'


class NHL:
    
    SEASON_TYPES = [ '02', '03' ]
    REG_GAME_CNT = 1271
    
    
    def __init__(self):
        self.rest = NHL_REST()
    
    def __add__(self):
        pass
    
    
    def get_data(self):
        pass
    
    def gen_season_id(self, year, game_type):
        return f'{year}{game_type}'
    
    def gen_game_id(self, year, game_type, game_cnt): 
        season_id = self.gen_season_id(year, game_type)
        return f'{season}{game_cnt}'
    