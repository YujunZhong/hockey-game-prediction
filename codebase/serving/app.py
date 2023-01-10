"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
import sys
from pathlib import Path

import logging
from markupsafe import escape
from flask import Flask, jsonify, request, abort
from flask_caching import Cache
from comet_client import CometClient

import sklearn
import pandas as pd
import pickle

#logging
def configure_logging(log_file):
    FORMAT = '%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
    logging.basicConfig(filename=log_file,
                level=logging.INFO, format=FORMAT)
    #strem
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(FORMAT)
    handler.setFormatter(formatter)
    root.addHandler(handler)

# caching
def setup_cache(): 
    cache_dir = '/tmp/nhl_models'
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    return Cache(config={"CACHE_TYPE": "filesystem",
            'CACHE_DIR': cache_dir,
            'CACHE_DEFAULT_TIMEOUT': 0}) # zero implies never timeout.


log_file = os.environ.get("FLASK_LOG", 'flask.log')
configure_logging(log_file)
cache = setup_cache()

# flask app initialisation
version = 'v3'
app = Flask(__name__)
cache.init_app(app)

@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # load default model
    default_model_name = 'best-hist-gradient-boosting'
    version = '1.0.0'
    #comet 
    WORKSPACE = 'ds-team-9'
    app.comet = CometClient(WORKSPACE, app.logger)
    model_path, description, model_name = app.comet.get_model("", default_model_name, version)
    model = pickle.load(open(model_path, 'rb'))
    response = f'Server is using the following model for prediction: {description}'
    app.logger.info(response)
    cache.set("server_model", [model_name, model])
    

@app.route("/")
def ping():
    message = f'NHL Analytics ({version}) is Active!!'
    app.logger.info(message)
    return message


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    with open(log_file) as log_info:
        response = log_info.read()

    return jsonify(response) # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here
    
    running_model = cache.get("server_model")
    
    model_path, description, name = app.comet.get_model(running_model[0], json['model_name'], json['version'])
    model = pickle.load(open(model_path, 'rb'))
    cache.set("server_model", [name, model])
    
    res = f'download_registry_model completed. Server is using the following model for prediction: {description}'
    response = {'description': res, 'server_model_name': name}
    app.logger.info(res)
    
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    running_model = cache.get("server_model")
    
    data_df = pd.DataFrame(json)
    prediction = running_model[1].predict(data_df.values)
    preds_prob = running_model[1].predict_proba(data_df.values)[:, 1] 
    prediction = prediction.tolist()
    preds_prob = preds_prob.tolist()
    
    res = f'Prediction: {prediction}'
    response = {'prediction': prediction, 'preds_prob': preds_prob}
    app.logger.info(res)
    
    return jsonify(response)  # response must be json serializable!
