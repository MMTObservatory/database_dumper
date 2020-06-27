from tornado.web import Application, RequestHandler
import json


import asyncio

import pandas as pd
import datetime

from ..db import db_conn
from ..appconfig import AppConfig
config=AppConfig()



class JsonHandler(RequestHandler):
    """This handler accepts a json string with the following keywords:
    ds_names: list of table names in measurements database to be queried
    delta_time: start query with timestamp > (now - delta_time)
    nsamples: the number of samples you want returned over the delta_time.
    if we are upsampling we will back fill the missing points otherwise we
    will take the mean.
    """

    conn = db_conn()

    async def get(self):
        params = json.loads(self.request.body.decode())
        info = await self.conn.info()
        good = []
        resp = {"errors": [], "info":[]}

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

        dataframes = {}
        for ds_name in good:
            df = await self.conn.recent_select(("value", "timestamp"), ds_name, delta_time)

            if len(df) == int(config["DEFAULT"]["resp_limit"]):
                # Too much data, let's not block for all of this.
                loop = asyncio.get_event_loop()
                fname = f"{ds_name}_{now.strftime('%m%d%H%M')}.csv"
                task = loop.create_task(self.conn.long_select(
                    ("value", "timestamp"),
                    ds_name,
                    delta_time,
                    fname=fname)
                )

                #handle = loop.call_soon(task)
                resp['info'].append(f"Over record limit, sending to {fname} soon.")

            df.index = pd.to_datetime(df.timestamp * 1000000)
            del df['timestamp']
            # period = delta_time.total_seconds()/nsamples
            # df = df.resample(datetime.timedelta(seconds=period)).mean().fillna(method='bfill')

            dataframes[ds_name] = df


        merged = pd.concat(dataframes, axis=1)
        #merged.interpolate()
        resp["data"] = json.loads(merged.to_json())
        resp["timestamp"] = timestamp
        resp["delta_time"] = delta_time.total_seconds()
        self.write(json.dumps(resp))


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


def make_app():
    urls = [("/recent/json", JsonHandler),
            ("/recent/get", JsonHandler),
            (r"/info(.*)", InfoHandler)]
    return Application(urls, debug=True)



