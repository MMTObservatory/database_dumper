FROM python:3.8-buster

LABEL maintainer="Scott Swindell <sswindell@mmto.org>"



WORKDIR /db_dumper


copy . /db_dumper


RUN apt update \
    && apt install -y automake gcc g++ python3-dev libffi-dev \
    && pip install wheel \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apk/* \
		&& cd /db_dumper \
		&&  pip install -r requirements.txt  \
		&& python setup.py install


CMD python /db_dumper/run.py
