FROM python:3.8-buster

LABEL maintainer="Scott Swindell <sswindell@mmto.org>"

ARG USER_ID
ARG GROUP_ID

RUN groupadd -f -g ${GROUP_ID} dbuser && \ 
	useradd -rm -d /home/dbuser -u ${USER_ID} -g ${GROUP_ID} -p foobar dbuser 

RUN mkdir /home/dbuser/notebooks && chown dbuser /home/dbuser/notebooks 


WORKDIR /db_dumper


copy . /build

RUN apt update \
    && apt install -y automake gcc g++ python3-dev libffi-dev \
    && pip install wheel \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apk/* \
	&& cd /build && pip install -r requirements.txt &&python setup.py install


USER dbuser

CMD python run.py
