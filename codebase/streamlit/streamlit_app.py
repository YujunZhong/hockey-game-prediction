import streamlit as st
import pandas as pd
import numpy as np

import os
import sys
import threading
import time
import signal

from ift6758.client import ServingClient, GameClient

# ===== load serving client ======
ip = 'serving'
port = '8890'
features = ['team', 'game_seconds', 'period', 'coord_x', 'coord_y', 'shot_distance', 'shot_angle', 'speed']
if "serving_client" not in st.session_state:
    st.session_state.serving_client = ServingClient(ip, port, features)

def preds_display(game_info, result, status='finished'):
    team_1 = game_info["Home_Team"]
    team_2 = game_info["Away_Team"]

    period = result['period'].values[-1]
    period_left_time = 1200 - (result['game_seconds'].values[-1] - (period - 1) * 1200)
    left_min = 0
    left_sec = 0
    if status=='live':
        left_min = period_left_time // 60
        left_sec = period_left_time % 60

    pred_goals_1 = result[result['team'] == team_1]['preds_prob'].sum()
    pred_goals_1 = round(pred_goals_1, 1)
    real_goals_1 = result[result['team'] == team_1]['label'].sum()
    diff1 = round(real_goals_1 - pred_goals_1, 1)

    pred_goals_2 = result[result['team'] == team_2]['preds_prob'].sum()
    pred_goals_2 = round(pred_goals_2, 1)
    real_goals_2 = result[result['team'] == team_2]['label'].sum()
    diff2 = round(real_goals_2 - pred_goals_2, 1)

    st.subheader(f"Game {game_id}: {team_1} vs {team_2}")
    st.text(f"Period {period} - {left_min}:{left_sec} left")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"{team_1} xG (actual)", value=f"{pred_goals_1}({real_goals_1})", delta=f"{diff1}")
    with col2:
        st.metric(label=f"{team_2} xG (actual)", value=f"{pred_goals_2}({real_goals_2})", delta=f"{diff2}")

    with st.container():
        # TODO: Add data used for predictions
        st.subheader("Data used for predictions (and predictions)")
        st.write(result)

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.title("Hockey Visualization App")

lr_models = ['logistic-regression-shot-angle', 'logistic-regression-shot-distance', 
            'logistic-regression-shot-distance-and-angle']

with st.sidebar:
    # TODO: Add input for the sidebar
    workspace = st.selectbox(
        'Workspace',
        ('ds-team-9',)
    )
    model_name = st.selectbox(
        'Model',
        ('XGBoost', 'best-cat-boost', 'best-hist-gradient-boosting', 
            'logistic-regression-shot-angle', 'logistic-regression-shot-distance', 
            'logistic-regression-shot-distance-and-angle')
    )
    version = st.selectbox(
        'Version',
        ('1.1.0', '1.0.0') if model_name in lr_models else ('1.0.0',)
    )

    if st.button('Get model'):
        st.session_state.serving_client.download_registry_model(workspace, model_name, version)
        st.text(f"Model loaded!")

with st.container():
    # TODO: Add Game ID input
    # ================================
    game_id = st.text_input('Game ID', '2022020485')

    game_client = GameClient()
    if game_client.get_game_status(game_id)[0] == 'FINAL':
        if st.button('Ping game'):
            with st.container():
                # TODO: Add Game info and predictions
                game_info = game_client.get_game_status(game_id)[1]
                game_df = game_client.get_game_df(game_id)
                result = st.session_state.serving_client.predict(game_df)
                preds_display(game_info, result, status='finished')

    elif game_client.get_game_status(game_id)[0] == 'live':
        res_features = features.copy()
        res_features.append('preds_prob')
        res_features.append('label')
        predicted_df = pd.DataFrame(columns=res_features)
        df_index = 0

        if st.button('Ping game'):
            with st.container():
                game_info = game_client.get_game_status(game_id)[1]
                game_df = game_client.get_game_df(game_id)
                new_game_df = game_df.loc[(game_df.index >= df_index)]    # filtering

                result_df = st.session_state.serving_client.predict(new_game_df)     # predict
                predicted_df = pd.concat([predicted_df, result_df], axis=0)
                #print(len(predicted_df))
                df_index = len(predicted_df)
                preds_display(game_info, predicted_df, status='live')

    else:
        if st.button('Ping game'):
            st.text("No data for this game yet")