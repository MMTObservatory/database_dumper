# -*- coding: utf-8 -*-

from .handlers import base
from .handlers import api
import tornado.web
from .appconfig import AppConfig
config = AppConfig()
from settings import settings

url_patterns = [
    (r'/', base.MainHandler),
    (r'/recent/json', api.JsonHandler),
    (r'/info', api.InfoHandler),
    (r'/info(.json)', api.InfoHandler),
    (r'/info(.html)', api.InfoHandler),
    (r'/info/system2log.json', api.System2LogHandler),
    (r'/job/data.html', api.JobInfoHandler),
    (r'/job/data.json', api.JobHandler),
    (r'/joblist(.html)', api.JobListHandler),
    (r'/joblist(.json)', api.JobListHandler),
    (r'/tasklist.html', api.TaskHandler),
    (r'/test/(.*)/(.*)', api.TestHandler),
    (r'/build-notebook', api.NotebookHandler),
    (r'/static/(.*)', tornado.web.StaticFileHandler),
    (r'/tmp/(.*)', tornado.web.StaticFileHandler, {"path": config["DEFAULT"]["bigfile_tmp_path"]}),
    (r'/data/(.*)', tornado.web.StaticFileHandler, {"path": config["DEFAULT"]["bigfile_path"]}),
    (r'/favicon.ico', tornado.web.StaticFileHandler, {"path": './static/assets/img/favicon.ico'})


]
