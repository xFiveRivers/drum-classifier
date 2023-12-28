FROM tensorflow/tensorflow:latest-gpu
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
RUN apt-get -y update \
    && apt-get -y install git