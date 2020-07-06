from tornado.web import Application, RequestHandler
import json


import asyncio

import pandas as pd
import datetime

from ..db import db_conn, dumper_job, job_interface
from ..appconfig import AppConfig
from dateutil import parser
from pathlib import Path
config = AppConfig()
from collections import OrderedDict


class JsonHandler(RequestHandler):
    """This handler accepts a json string with the following keywords:
    ds_names: list of table names in measurements database to be queried
    delta_time: start query with timestamp > (now - delta_time)
    nsamples: the number of samples you want returned over the delta_time.
    if we are upsampling we will back fill the missing points otherwise we
    will take the mean.
    """

    conn = db_conn()
    jiface = job_interface()


    async def post(self):

        resp = {"errors": [], "info": [], "success":False}


        try:
            params = json.loads(self.request.body.decode())
        except Exception as error:
            self.write(resp["errors"].append(str(error)))
            return

        info = await self.conn.info()
        good = []


        for name in params["ds_names"]:
            if any(info.ds_name == name):
                good.append(name)
            else:
                resp["errors"].append(f"Bad ds_name {name}")

        try:
            start = parser.parse(params["startdate"])
        except Exception as err:
            resp["errors"].append(str(err))
            start=None

        try:
            stop = parser.parse(params["enddate"])
        except Exception as err:
            resp["errors"].append(str(err))
            stop = None

        if len(good) == 0:
            self.write(json.dumps(resp))
            return

        try:
            nsamples = int(params["nsamples"])
        except Exception as err:
            resp["errors"].append(str(err))
            nsamples = 100000

        try:
            fit_order = int(params["fit_order"])
        except Exception as err:
            resp["errors"].append(str(err))
            fit_order = 5

        job = dumper_job(good, nsamples, fit_order, start, stop)
        job = dumper_job(good, nsamples, fit_order, start, stop)
        self.jiface.submit_job(job)
        resp["info"] = job.state
        resp["success"] = True

        self.write(json.dumps(resp, default=str))
        self.set_header("Content-Type", "application/json")

class TestHandler(RequestHandler):

    def get(self, *args):

        self.write(f"This is a test {args[0]} {args[1]}")

class JobHandler(RequestHandler):

    jiface = job_interface()

    def get(self):

        uuid = self.get_argument('jobid')
        job = self.jiface[uuid]
        jsondata = json.dumps(job.state, default=str)
        self.write(jsondata)
        self.set_header("Content-Type", "application/json")



class JobInfoHandler(RequestHandler):

    jiface = job_interface()

    def get(self):
        jobid = self.get_argument("jobid")
        self.render(
            'jobs.html',
            jobid=jobid, job=self.jiface[jobid].state
        )



class JobListHandler(RequestHandler):

    jiface = job_interface()

    def get(self, type):
        if type == ".json":
            self.write({'ids':list(self.jiface.iterids())})
        else:
            self.render(
                'joblist.html',
                jobs=self.jiface
            )


class GetHandler(RequestHandler):
    conn = db_conn()

    def get(self):
        self.write({'message': str(self.request.body)})


class InfoHandler(RequestHandler):
    conn = db_conn()

    async def get(self, otype='.html'):
        print(otype)
        df = await self.conn.info()
        if otype == '.html':
            self.write(df.to_html())

        else:
            self.write(df.to_json())

class System2LogHandler(RequestHandler):
    conn = db_conn()

    async def get(self):
        try:
            df = await self.conn.info()
        except Exception as error:
            self.write({"error": "Could not connect to database"})
            return
        systems = set(df['subsystem'])
        output = OrderedDict()

        for sys in systems:
            output[sys] = list(df[df['subsystem']==sys]['ds_name'])

        self.write(output)



def make_app():
    urls = [("/recent/json", JsonHandler),
            ("/recent/get", JsonHandler),
            (r"/info(.*)", InfoHandler)]
    return Application(urls, debug=True)



