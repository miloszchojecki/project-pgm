FROM python:3.11-bullseye

EXPOSE 4001

WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential git \
    pip install --upgrade pip wheel setuptools

COPY requirements.txt /app

RUN pip install -r requirements.txt 


