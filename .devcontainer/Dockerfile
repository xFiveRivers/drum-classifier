FROM tensorflow/tensorflow:latest-gpu
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
RUN apt -y update
RUN apt -y upgrade
RUN apt-get -y install git