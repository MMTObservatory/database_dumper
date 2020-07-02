# -*- coding: utf-8 -*-

from .handlers import base
from .handlers import api
import tornado.web
from settings import settings

url_patterns = [
    (r'/', base.MainHandler),
    (r'/recent/json', api.JsonHandler),
    (r'/info', api.InfoHandler),
    (r'/info(.json)', api.InfoHandler),
    (r'/info(.html)', api.InfoHandler),
    (r'/info/system2log.json', api.System2LogHandler),
    (r'/job/info.html', api.JobInfoHandler),
    (r'/job/data.json', api.JobHandler),
    (r'/test/(.*)/(.*)', api.TestHandler),
    (r'/static/(.*)', tornado.web.StaticFileHandler)



]
