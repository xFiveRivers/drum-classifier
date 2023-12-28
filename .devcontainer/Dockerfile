FROM tensorflow/tensorflow:latest-gpu
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
RUN apt update
RUN apt upgrade
RUN apt-get install git