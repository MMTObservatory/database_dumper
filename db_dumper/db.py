import pandas as pd
import aiomysql
import logging
import datetime
from pathlib import Path
from .appconfig import AppConfig
from uuid import uuid1
import asyncio
import numpy as np


global_limit = int(AppConfig()["DEFAULT"]["resp_limit"])


class db_conn:
    instance = None

    class _db_conn:

        def __init__(self):
            self.config = AppConfig()

            self.cache = {}

        async def info(self):
            if "info" not in self.cache:
                self.cache["info"] = await self.select(
                    ["ds_name", "description", "legendtext", "system", "subsystem"],
                    "aaa_parameters")

            return self.cache["info"]

        async def select(self, columns, table, where=None):
            colstr = columns[0]
            for col in columns[1:]:
                colstr += ', ' + col

            sqlstr = f"SELECT {colstr} FROM {table}"
            if where:
                sqlstr += f" where {where}"

            sqlstr += f" limit {global_limit}"
            print(sqlstr)
            resp = await self.query(sqlstr)

            df = pd.DataFrame.from_records(resp, columns=columns)
            return df

        async def recent_select(self,
                                columns: list,
                                table: str,
                                recency=None,
                                start=None,  # not implemented
                                stop=None  # not implemented
                                ):

            """SQL select from one of the measurements sensor tables
            This query assumes the tables primary key is timestamp
            is an integer in milliseconds.
            """
            colstr = columns[0]
            for col in columns[1:]:
                colstr += ', ' + col

            now = datetime.datetime.utcnow()
            earlier = now - recency
            where = f"timestamp > {earlier.timestamp() * 1000}"
            sqlstr = f"SELECT {colstr} FROM {table}"
            if where:
                sqlstr += f" where {where}"

            sqlstr += f" limit {global_limit}"
            print(sqlstr)
            resp = await self.query(sqlstr)

            df = pd.DataFrame.from_records(resp, columns=columns)

            return df

        async def long_select(self, table, jobid, fname, start, stop):
            """This function is meant to be called when the number of
            records that will be retrieved by a select is greater than
            some upper limit. In this case we call the select in a
            loop and write the responses to a file allowing other
            coroutines to advance during the IO blocking."""

            # now = datetime.datetime.utcnow()
            #
            # if fname is None:
            #     fname = f"{table}_{now.strftime('%m%d%H%M')}.csv"
            #
            # fpath = Path(self.config["DEFAULT"]["bigfile_tmp_path"]) / jobid
            # fpath.mkdir(parents=True, exist_ok=True)

            where = f"timestamp > {int(start.timestamp() * 1000)}\
             and timestamp < {int(stop.timestamp() * 1000)}"

            sqlstr = f"SELECT timestamp, value FROM {table}"
            if where:
                sqlstr += f" where {where}"

            n_resp = global_limit

            cursor = await self.query(sqlstr, return_type="cursor")

            while n_resp >= global_limit:
                records = await cursor.fetchmany(global_limit)
                n_resp = len(records)
                yield records




                # print(f"Writing {len(df)} records to {fname}")
                # if is_first:
                #     # Right the header the first
                #     # iteration but not subsequent iterations
                #     header = True
                #     is_first = False
                # else:
                #     header = False
                #
                # df.to_csv(fpath / fname, mode='a', header=header)
                #
                # del df



        async def query(self, sql: str, return_type: str = "records"):
            logging.debug(f'config is {dict(self.config["DEFAULT"])}')
            conn = await \
                aiomysql.connect(
                    host=self.config["DEFAULT"]['mysql_host'],
                    user=self.config["DEFAULT"]['mysql_user'],
                    password=self.config["DEFAULT"]['mysql_pass'],
                    db=self.config["DEFAULT"]["mysql_dbname"],
                    port=int(self.config["DEFAULT"]["mysql_port"]))

            cur = await conn.cursor()
            await cur.execute(sql)
            if return_type == "records":
                resp = await cur.fetchall()
            else:
                resp = cur

            return resp

    def __new__(cls):
        if not db_conn.instance:
            db_conn.instance = db_conn._db_conn()
        return db_conn.instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, key, value):
        return setattr(self.instance, key, value)


class dumper_job:
    conn = db_conn()

    def __init__(self, ds_names: list, recency: int, nsamples: int=1000):
        if not isinstance(recency, datetime.timedelta):
            recency = datetime.timedelta(seconds=recency)
        self.config = AppConfig()
        self.ds_names = ds_names
        self.recency = recency
        self.stop = datetime.datetime.utcnow()
        self.start = self.stop - recency
        self.chunksize = 100000
        self.jobid = str(uuid1())
        self.nsamples = nsamples

        self.metadata = {"job_start_utc": datetime.datetime.utcnow()}

    async def run(self):

        tasks = []

        # Set up the sql select queries
        for table in self.ds_names:
            select_iter = self.conn.long_select(table, self.jobid, None, self.start, self.stop)
            task = self.collect_and_write(table, select_iter)
            self.metadata[table] = {"written": False}
            tasks.append(task)

        # Get the raw data
        results = await asyncio.gather(*tasks)

        # downsample the data
        return [self.down_sample(result) for result in results]


    async def collect_and_write(self, table, select_iter):

        fpath = Path(self.config["DEFAULT"]["bigfile_tmp_path"]) / self.jobid
        fpath.mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.utcnow()
        fname = f"{table}_{now.strftime('%m%d%H%M')}.csv"

        self.metadata[table]["tmpfile"] = str(fpath/fname)

        is_first = True
        async for records in select_iter:

            dataframe = pd.DataFrame.from_records(
                records,
                columns=("timestamp", "value"),
                index="timestamp")

            dataframe.to_csv(fpath / fname, header=is_first, mode='a')
            if is_first:  # only write header on first iteration
                is_first = False

        del dataframe
        self.metadata[table]["written"] = True
        return fpath/fname

    def __repr__(self):

        return str(self.state)


    @property
    def state(self):
        resp = self.metadata
        resp["t0_utc"] = self.start
        resp["tf_utc"] = self.stop
        resp["tspan_hours"] = (self.stop - self.start).total_seconds() / 3600

        return resp

    @property
    def sample_times(self):
        return pd.date_range(self.start, self.stop, periods=self.nsamples)

    def down_sample(self, csv_file):
        chunks = pd.read_csv(csv_file, chunksize=10000)
        sample_times = self.sample_times
        out_df = pd.DataFrame()

        for ii, df in enumerate(chunks):
            print(ii)
            #yield df
            df.index = pd.to_datetime(df.timestamp * 1000000)  # Pandas dt is in nanoseconds
            del df["timestamp"]
            subsample = sample_times[sample_times < df.index.max()]
            subsample = subsample[subsample > df.index.min()]
            with_sample_times = pd.concat([df, pd.DataFrame([np.nan] * len(subsample), index=subsample)])
            newdf = with_sample_times.value.interpolate()[subsample]

            out_df = pd.concat([out_df, newdf])
            print(f"{ii}th iteration frame length = {len(out_df)}")
            #yield {"out_df": out_df, "newdf": newdf, "sample_times": sample_times, "subsample": subsample}

        return out_df
