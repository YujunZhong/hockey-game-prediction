import os
import json
import requests
import pandas as pd
import logging

import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('client.log')
logger.addHandler(file_handler)

class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")
        
        if features is None:
            features = ["shot_distance"]
        self.features = features
        
        # any other potential initialization
        self.model_name = 'best-hist-gradient-boosting'

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        
        if (self.model_name=='XGBoost'):
            feat_list = ['game_seconds', 'period', 'coord_x', 'coord_y', 'shot_distance', 'shot_angle', 'shot_type', 'prev_event_type', 'prev_coord_x', 'prev_coord_y', 'time_from_prev_event', 
            'distance_from_prev_event', 'rebound', 'change_in_angle', 'speed', 'empty_net']
        elif (self.model_name=='best-hist-gradient-boosting'):
            feat_list = ['game_seconds', 'period', 'coord_x', 'coord_y', 'shot_distance', 'shot_angle', 'shot_type', 'prev_event_type', 'prev_coord_x', 'prev_coord_y', 
            'time_from_prev_event', 'distance_from_prev_event', 'rebound', 'change_in_angle', 'speed', 'empty_net', 'is_home', 'is_forward', 'is_shortHanded']
        elif (self.model_name=='logistic-regression-shot-angle'):
            feat_list = ['shot_angle']
        elif (self.model_name=='logistic-regression-shot-distance'):
            feat_list = ['shot_distance']
        elif (self.model_name=='logistic-regression-shot-distance-and-angle'):
            feat_list = ['shot_distance', 'shot_angle']
        else:
            feat_list = ['game_seconds', 'period', 'coord_x', 'coord_y', 'shot_distance', 'shot_angle', 'shot_type', 'prev_event_type', 'prev_coord_x', 'prev_coord_y', 
            'time_from_prev_event', 'distance_from_prev_event', 'rebound', 'change_in_angle', 'speed', 'empty_net', 'is_home', 'is_forward', 'is_shortHanded']
        
        test_data = X[feat_list]
        label = X['event_type'].tolist()

        r = requests.post(
            f"{self.base_url}/predict", 
            json=json.loads(test_data.to_json())
        )
        
        res = r.json()
        
        df_result = X[self.features]

        df_result['prediction'] = res['prediction']
        df_result['preds_prob'] = res['preds_prob']
        df_result['label'] = label

        return df_result
    

    def logs(self) -> dict:
        """Get server logs"""
        
        r = requests.get(f"{self.base_url}/logs")
        result = r.json()

        return result

    def download_registry_model(self, workspace: str, model_name: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """

        req = {'workspace': workspace, 'model_name': model_name, 'version': version}
                
        r = requests.post(
            f"{self.base_url}/download_registry_model", 
            json=json.loads(json.dumps(req))
        )
        
        res = r.json()
        self.model_name = res['server_model_name']
        result = {'workspace': workspace, 'model_name': res['server_model_name'], 'version': version}
        
        return result
    
