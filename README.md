# Hockey Game Prediction
This repository includes three parts for hockey game prediction: model deployment, live workflows and interactive visualization. The training code for the models and the visualization of the NHL data are in other branches.


## Setup & Deployment Guide

Applications to run:

* Serving (Flask)
* Streamlit

### Pre-requisite

* Users should have `docker` installed in their machine. 
* Store your Comet ML API key in `COMET_API_KEY` environment variable.

Both these applications are can be easily deployed using dockers.

### Method 1:

1) Execute `build.sh` (shell script) to build the docker images of serving and streamlit application.

2) Execute `run.sh` (shell script) to run the serving and streamlit containers using their respective docker images.

### Method 2: 

* Run `compose.sh` (shell script) to build and run docker containers. This a single step process which uses **docker-compose**. Users can configure PORT and other variables via `compose.env`. 

Default configuration:
```
serving   - 8890(PORT) 
streamlit - 8880(PORT)
```

### Usage

Users can access the application using these URLs: [localhost:8880](http://localhost:8880) or [127.0.0.1:8880](http://127.0.0.1:8880)
