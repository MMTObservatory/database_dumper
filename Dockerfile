FROM python:3.8-alpine

LABEL maintainer="Scott Swindell <sswindell@mmto.org>"

USER root

WORKDIR /app
ADD . /app/

RUN apk --update add --no-cache git \
    && pip install wheel \
    && pip install -r requirements.txt \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apk/*
