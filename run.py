#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Basic run script"""

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.autoreload
from tornado.options import options
import tornado.web
import os
from db_dumper.appconfig import AppConfig 

if "DB_DUMPER_CONFIG" in os.environ:
    configpath = os.environ["DB_DUMPER_CONFIG"]
else:
    configpath = "config.ini"

config = AppConfig(configpath)

from settings import settings
from db_dumper.urls import url_patterns
import logging

logging.basicConfig(level=logging.DEBUG)

class TornadoApplication(tornado.web.Application):

    def __init__(self):
        tornado.web.Application.__init__(self, url_patterns, **settings)


def main():

    app = TornadoApplication()
    app.listen(config["DEFAULT"]["app_port"])
    logging.debug(dict(config["DEFAULT"]))
    logging.debug(f"port is {config['DEFAULT']['app_port']}")
    loop = tornado.ioloop.IOLoop.current()

    loop.start()


if __name__ == "__main__":
    main()
