from tornado.web import Application, RequestHandler
import json


import asyncio

import pandas as pd
import datetime

from ..db import db_conn, dumper_job, job_interface
from ..appconfig import AppConfig
from pathlib import Path
config = AppConfig()



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

    # async def get(self):
    #     try:
    #         params = json.loads(self.request.body.decode())
    #     except Exception as error:
    #         self.write(str(error))
    #         return
    #     info = await self.conn.info()
    #     good = []
    #     resp = {"errors": [], "info":[]}
    #
    #     for name in params["ds_names"]:
    #         if any(info.ds_name == name):
    #             good.append(name)
    #         else:
    #             resp["errors"].append(f"Bad ds_name {name}")
    #
    #     try:
    #         delta_time = datetime.timedelta(seconds=int(params["delta_time"]))
    #     except Exception as err:
    #         resp["errors"].append(str(err))
    #         # default to 1 hour
    #         delta_time = datetime.timedelta(seconds=3600)
    #
    #     now = datetime.datetime.utcnow()
    #     timestamp = int((now - delta_time).timestamp() * 1000)  # unix timestamp in milliseconds
    #
    #     if len(good) == 0:
    #         self.write(json.dumps(resp))
    #         return
    #
    #     try:
    #         nsamples = int(params["nsamples"])
    #     except Exception as err:
    #         resp["errors"].append(str(err))
    #         nsamples = 100000
    #
    #     try:
    #         fit_order = int(params["fit_order"])
    #     except Exception as err:
    #         resp["errors"].append(str(err))
    #         fit_order = 3
    #
    #     job = dumper_job(good, delta_time, nsamples, fit_order)
    #     self.jiface.submit_job(job)
    #     self.write(self.jiface.state)


    async def post(self):

        resp = {"errors": [], "info": [], "success":False}
        jobid = self.get_secure_cookie("job")

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
            delta_time = datetime.timedelta(seconds=int(params["delta_time"]))
        except Exception as err:
            resp["errors"].append(str(err))
            # default to 1 hour
            delta_time = datetime.timedelta(seconds=3600)

        now = datetime.datetime.utcnow()
        timestamp = int((now - delta_time).timestamp() * 1000)  # unix timestamp in milliseconds

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
            fit_order = 3

        job = dumper_job(good, delta_time, nsamples, fit_order)
        self.jiface.submit_job(job)
        resp["info"] = self.jiface.state
        resp["success"] = True
        self.set_cookie("job", job.jobid)
        self.write(resp)

class TestHandler(RequestHandler):

    def get(self, *args):

        self.write(f"This is a test {args[0]} {args[1]}")

class JobHandler(RequestHandler):

    jiface = job_interface()

    def get(self, do_what):
        # self.render(
        #     r'C:\Users\srswi\git-clones\db_dumper\db_dumper\templates\jobs.html',
        #     job=self.jiface[uuid]['job'])
        jobid = self.get_argument("jobid")

        self.render(r'C:\Users\srswi\git-clones\db_dumper\db_dumper\templates\jobs.html')


class JobInfoHandler(RequestHandler):

    jiface = job_interface()

    def get(self, do_what):
        # self.render(
        #     r'C:\Users\srswi\git-clones\db_dumper\db_dumper\templates\jobs.html',
        #     job=self.jiface[uuid]['job'])

        self.write(self.jiface[jobid].state)

class JobListHandler(RequestHandler):

    jiface = job_interface()

    def get(self, do_what):
        self.render(
            r'C:\Users\srswi\git-clones\db_dumper\db_dumper\templates\joblist.html',
            self.jiface()

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
        output = {}
        for sys in systems:
            output[sys] = list(df[df['subsystem']==sys]['ds_name'])
            #output[sys]= sys

        self.write(output)



def make_app():
    urls = [("/recent/json", JsonHandler),
            ("/recent/get", JsonHandler),
            (r"/info(.*)", InfoHandler)]
    return Application(urls, debug=True)



