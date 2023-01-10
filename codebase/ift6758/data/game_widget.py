import json
import ipywidgets as widgets
from IPython.display import display
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

SEASON_YEAR = 'season_year'
GAME_ID = 'game_id'
GAME_TYPE = 'game_type'
map_game_type = { 'R': 'Regular', 'P': 'Playoff' }
START_TIME = 'start_time'
EVENTS = 'events'
EVENT_INFO = 'event_info'
PERIOD = 'period'
PERIOD_TIME = 'periodTime'
TEAM_NAME = 'name'
GOALS = 'goals'
SOG = 'SoG'

HOME = 'home'
HOME_GOALS = 'home_goals'
HOME_SOG = 'home_sog'

AWAY = 'away'
AWAY_GOALS = 'away_goals'
AWAY_SOG = 'away_sog'

GAME_INFO_TEMPLATE = 'game_info.template'
EVENT_INFO_TEMPLATE = 'event_info.template'

ICE_RINK_IMG_PATH = './figures/nhl_rink.png'



def populate_team(team_info):
    '''
    Utility method to populate the team info as a dict.
    :param team_info: dict (obtained from game object of nhl raw_data)
    '''
    team = dict()
    team[TEAM_NAME] = team_info['team']['triCode']
    team[GOALS] = team_info['goals']
    team[SOG] = team_info['shotsOnGoal']
    return team
    
def clean_game_data(game_data):
    '''
    Utility method to clean the game data.
    :param game_data: dict (game object of nhl raw_data)
    '''
    game, liveData = game_data['gameData'], game_data['liveData']
    c_game = dict()
    season_year = f"{game['game']['season'][:4]}-{game['game']['season'][4:]}"
    c_game[SEASON_YEAR] = season_year
    c_game[GAME_ID] = int(game['game']['pk'])
    c_game[GAME_TYPE] = game['game']['type']
    c_game[START_TIME] = game['datetime']['dateTime']
    c_game[EVENTS] = liveData['plays']['allPlays']
    c_game[HOME] = populate_team(liveData['linescore']['teams'][HOME])
    c_game[AWAY] = populate_team(liveData['linescore']['teams'][AWAY])
    return c_game

def get_pbp_data(season):
    '''
    Utility method to get play-play-play data for all the games in the season.
    :param season: dict (obtained from nhl raw_data)
    '''
    c_season = list(map(clean_game_data, season))
    c_regular, c_playoff = [], []
    for g in c_season:
        (c_playoff, c_regular)[g[GAME_TYPE] == 'R'].append(g)
    c_regular.sort(key=lambda g: g[GAME_ID])
    c_playoff.sort(key=lambda g: g[GAME_ID])
    return c_regular, c_playoff

def load_template(path):
    '''
    Utility  method to load the content display template for the widget
    :param path: string path
    '''
    f = open(path, encoding = 'utf-8')
    template = f.read()
    return template

class IceRink:
    '''
    IceRink stores the dimensions and edge co-ordinates of the ice-rink.
    '''
    length = 200
    width = 85
    x_min = -length/2
    x_max = length/2
    y_min = -width/2
    y_max = width/2
    x_locs = 8
    y_locs = 4
    
class PlotEvent:
    '''
    PlotEvent generates the visual representation of the play on the ice-rink by co-ordinates of the play
    and the type of play
    '''

    marker_map = {'FACEFOFF': 'o', 'SHOT' : 'p', 'GOAL': '*', 'GIVEAWAY': 'X',
                'HIT' : 'D', 'TAKEAWAY': 's', 'FIGHT' : 'X', 'PENALTY' : 'P', 'DEFAULT': 'o'}
    
    ir = IceRink()
    
    def __init__(self, ice_rink_path):
        self.irmg = mpimg.imread(ice_rink_path)
        plt.style.use('default')
    
    def plot(self, event, home, away):
        '''
        Plots the event play on the ice-rink using matplolib.
        :param event: event infomation (dict)
        :param home: name of the home team
        :param away: name of the away team
        '''
        play_event = False
        
        #check if its a play event
        p = event['coordinates']
        if any(p):
            play_event = True
            # Find the co-ordinates (x,y) on image
            x, y = p['x'], p['y']
        
        # the title and description
        title = self.get_title(event, home, away)
        event_type = event['result']['eventTypeId']
        # marker based on event_type
        marker = self.get_marker(event_type)
        # xy-axis label
        axis_label = 'feet' # same for all
        #upper_axis_label = 'team1' + '\t' * 5 + 'team2'
        
        
        
        # clear previous plots
        #plt.clf()
        
        # plot fresh one here
        plt.imshow(self.irmg, origin = 'upper', 
                   extent = (self.ir.x_min, self.ir.x_max, self.ir.y_min, self.ir.y_max))
        
        # axis labels
        plt.xlabel(axis_label)
        plt.ylabel(axis_label)
        #ax.twiny().set_xlabel(upper_axis_label) #buggy messes up image

        # plot title (event description)
        plt.title(title)
        
        # plot the co-ordinates
        if play_event:
            plt.plot([x], [y], marker=marker, ms = '10', color='blue')
            
    def get_title(self, event, home, away):
        description = event['result']['description']
        event_type = event['result']['eventTypeId']
        period = event['about'][PERIOD]
        period_time = event['about'][PERIOD_TIME]
        left, right = (home, away) if int(period) % 2 else (away, home)
        t_line_1 = f'{event_type} - {description}'
        t_line_2 = f'{period_time} P-{period}'
        t_line_3 = left + ' ' * 31 +  right
        title = f'{t_line_1}\n{t_line_2}\n{t_line_3}'
        return title
        
    def get_marker(self, event_type):
        key = event_type if event_type in self.marker_map else 'DEFAULT'
        return self.marker_map[key]
        
class PlayByPlayWidget:
    '''
    Widget to display play-by-play information of an NHL games in a season.
    '''
    game_info_template = load_template(GAME_INFO_TEMPLATE)
    event_info_template = load_template(EVENT_INFO_TEMPLATE)

    def __init__(self, season):
        self.regular, self.playoffs = get_pbp_data(season)
        self.games = self.regular
        self.plot_event = PlotEvent(ICE_RINK_IMG_PATH)
        self.game_info = widgets.HTML()
        self.event_info = widgets.HTML()
        self.__initialize_widgets()
        self.update_game_info(self.game.value)
        
    def __initialize_widgets(self):
        '''
        Initializes the widgets which composes of 2 int sliders - one for game and another for event.
        Then html widgets to dipslay relavent content and the ice-rink.
        '''
        self.season = widgets.Dropdown( options=[('Regular', False), ('Playoffs', True)],
                                       value=False,
                                       description='Season:', disabled=False)
        self.game = widgets.IntSlider(value=1, min = 1, max=len(self.games),
                                         description = "Game", continuous_update=False)
        self.event_slider = widgets.IntSlider(value = 1, min = 1,
                                          description = "Event", continuous_update=False)
        
        def handle_season_change(change):
            season_type = change.new
            print(f"Season Regular: {season_type}")
            if season_type == True:
                self.games = self.playoffs
            else:
                self.games = self.regular
                
            self.game.max = len(self.games)
            self.game.value = 1
            self.update_game_info(self.game.value)
        
        def handle_game_change(change):
            game_no = change.new
            print(f"Game value change: {game_no}")
            #update game values
            self.update_game_info(game_no)
             
        def handle_event_change(event_no):
            self.update_event_info(self.game.value, event_no)
        
        self.season.observe(handle_season_change, names='value')
        self.game.observe(handle_game_change, names='value')
        self.event = widgets.interactive(handle_event_change, event_no = self.event_slider)
        #print(help(self.event))

    def display(self):
        '''
        Displays all the widgets.
        '''
        hbox = widgets.HBox([self.season, self.game])
        w = widgets.VBox([hbox, self.game_info, self.event, self.event_info])
        display(w)
        
    def update_game_info(self, game_no):
        '''
        Update game_info based on the number on the game_slider.
        :param game_no: int
        '''
        game = self.games[game_no - 1]
        season_year = game[SEASON_YEAR]
        game_id = game[GAME_ID]
        game_type = map_game_type[game[GAME_TYPE]]
        start_time = game[START_TIME]
        home = game[HOME][TEAM_NAME]
        home_goals = game[HOME][GOALS]
        home_sog = game[HOME][SOG]
        
        away = game[AWAY][TEAM_NAME]
        away_goals = game[AWAY][GOALS]
        away_sog = game[AWAY][SOG]
        
        params = [(SEASON_YEAR, season_year), (GAME_ID, game_id), (GAME_TYPE, game_type), (START_TIME, start_time), (HOME, home), (AWAY, away), (HOME_GOALS, home_goals), (AWAY_GOALS, away_goals), (HOME_SOG, home_sog), (AWAY_SOG, away_sog)]
        
        html = self.game_info_template.format(**dict(params))
        self.game_info.value = html
        self.event_slider.value = 1
        self.event_slider.max =  max(1, len(self.games[game_no - 1][EVENTS]))
        self.event.update(self.event_slider.value)
    
    def update_event_info(self, game_no, event_no):
        '''
        Update event based on the number on the game_slider and event_slider.
        :param game_no: int
        :param event_no: int
        '''
        game = self.games[game_no - 1]
        events = game[EVENTS]
        if len(events) > 0:
            event = events[event_no - 1]
            home = game[HOME][TEAM_NAME]
            away = game[AWAY][TEAM_NAME]
            self.plot_event.plot(event, home, away)
            params = [(EVENT_INFO, json.dumps(event, sort_keys=True, indent=4))]
            html = self.event_info_template.format(**dict(params))
            
        else:
            html = "No events present!"
        self.event_info.value = html

