# -*- coding: utf-8 -*-

from .handlers import base
from .handlers import api


url_patterns = [
    (r'/', base.MainHandler),
    (r'/recent/json', api.JsonHandler),
    (r'/info', api.InfoHandler),
    (r'/info(.json)', api.InfoHandler),
    (r'/info(.html)', api.InfoHandler)

]
