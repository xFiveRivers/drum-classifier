FROM tensorflow/tensorflow:2.14.0-gpu
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
RUN apt-get -y update \
    && apt-get -y install git