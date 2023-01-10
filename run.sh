	#!/bin/bash

echo "Run docker containers"

docker run -d -p 8890:8890 --ip 0.0.0.0 --name serving --env COMET_API_KEY=$COMET_API_KEY ift6758/serving:1.0.0
docker run -d -p 8880:8880 --ip 127.0.0.1 --name streamlit ift6758/streamlit:1.0.0