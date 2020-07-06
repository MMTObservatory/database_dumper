FROM python:3.8-buster

LABEL maintainer="Scott Swindell <sswindell@mmto.org>"

USER root

WORKDIR /db_dumper

RUN apt update \
    && apt install -y automake gcc g++ python3-dev libffi-dev \
    && pip install wheel \
    && pip install -r requirements.txt \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apk/*

CMD python run.py
