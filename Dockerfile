# This dockerfile uses multi-stage build feature to 
# define both docker images. So, there is no need
# of two separate dockerfiles.

###########   SERVING ##################

FROM python:3.10-slim as serving
WORKDIR /code
ADD ./codebase/serving .
RUN pip install -r requirements.txt

ARG PORT=8890
ENV PORT_NUMBER=$PORT
ENV COMET_API_KEY='comet_api_key'

EXPOSE ${PORT_NUMBER}
CMD waitress-serve --listen=0.0.0.0:${PORT_NUMBER} app:app


###########   STREAMLIT ##################

FROM python:3.10-slim as streamlit

WORKDIR /code
ADD ./codebase/ift6758 ./ift6758
ADD ./codebase/streamlit  ./codebase/setup.py  ./

RUN pip install -r requirements.txt
RUN pip install -e .

ARG PORT=8880
ENV PORT_NUMBER=$PORT

EXPOSE ${PORT_NUMBER}
CMD streamlit run streamlit_app.py --server.port=${PORT_NUMBER} 
