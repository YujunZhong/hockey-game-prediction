#!/bin/bash

echo "Building Docker Images"

docker build --platform linux/amd64 --target serving -t ift6758/serving:1.0.0 --build-arg PORT=8890 -f Dockerfile .
docker build --platform linux/amd64 --target streamlit -t ift6758/streamlit:1.0.0 --build-arg PORT=8880 -f Dockerfile .