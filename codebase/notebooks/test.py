import os
import sys
import numpy as np 
import threading
import time
import signal
import pandas as pd

module_path = os.path.abspath("../")
sys.path.append(module_path)
module_path = os.path.abspath("../ift6758/client")
sys.path.append(module_path)

from serving_client import ServingClient
from game_client import GameClient


game_id = 2022020485

exit_event = threading.Event()

def get_live_data(game_id, features):
    res_features = features
    res_features.append('prediction')
    res_features.append('label')
    predicted_df = pd.DataFrame(columns=res_features)
    df_index = 0
    while not game_client.get_game_status(game_id)[0] == 'FINAL':
        game_df = game_client.get_game_df(game_id)
        result_df = game_df.loc[(game_df.index >= df_index)]
        predicted_df = pd.concat([predicted_df, result_df], axis=0)
        #print(len(predicted_df))
        df_index = len(predicted_df)
        print(predicted_df)
        time.sleep(60)
        
        if exit_event.is_set():
            break
            
def signal_handler(signum, frame):
    exit_event.set()
            
def signal_handler(signum, frame):
    exit_event.set()

ip = '127.0.0.1'
port = '30000'
            
workspace = 'ds-team-9'
# model_name -> 'XGBoost', 'best-hist-gradient-boosting', 'logistic-regression-shot-angle', 
#               'logistic-regression-shot-distance', 'logistic-regression-shot-distance-and-angle'
model_name = 'XGBoost'
version = '1.0.0'

features = ['game_seconds', 'period', 'coord_x', 'coord_y', 'shot_distance']

game_client = GameClient()
serving_client = ServingClient(ip, port, features)

serving_client.download_registry_model(workspace, model_name, version)

if game_client.get_game_status(game_id)[0] == 'FINAL':
    game_df = game_client.get_game_df(game_id)
    result = serving_client.predict(game_df)
    print(result)
elif game_client.get_game_status(game_id)[0] == 'Preview':
    print("Game not started yet")
elif game_client.get_game_status(game_id)[0] == 'live':
    signal.signal(signal.SIGINT, signal_handler)
    th = threading.Thread(target=get_live_data, args=(game_id,))
    th.start()
    th.join()
    
