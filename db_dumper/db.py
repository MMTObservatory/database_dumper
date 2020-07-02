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

        async def long_select(self, table, start, stop):
            """This function is meant to be called when the number of
            records that will be retrieved by a select is greater than
            some upper limit. In this case we call the select in a
            loop and write the responses to a file allowing other
            coroutines to advance during the IO blocking."""

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

        async def query(self, sql: str, return_type: str = "records"):

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

    def __init__(self, ds_names: list, recency: int, nsamples: int = 1000, fit_order: int=3):
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
        self.fit_order = fit_order

        self.metadata = {"job_start_utc": datetime.datetime.utcnow(), 'tables': {}}

        for table in self.ds_names:
            self.metadata['tables'][table] = {}
            self.metadata['tables'][table]['tmpfile'] = None
            self.metadata['tables'][table]['rows_written'] = 0

    async def run(self):
        """Retrieve raw data from the database, downsample data into evenly
        spaced datapoints and merge the data into one csv_file"""
        tasks = []
        fpath = Path(self.config["DEFAULT"]["bigfile_path"]) / self.jobid
        fpath.mkdir(exist_ok=True, parents=True)
        fname = "processed.csv"
        # Set up the sql select queries
        for table in self.ds_names:
            select_iter = self.conn.long_select(table, self.start, self.stop)
            task = self.collect_and_write(table, select_iter)
            self.metadata['tables'][table]['finished'] = True
            tasks.append(task)

        # Get the raw data
        results = await asyncio.gather(*tasks)

        # downsample the data
        dfs = [self.down_sample(result, table) for result, table in zip(results, self.ds_names)]
        final_df = pd.concat(dfs, join='outer', axis=1)
        final_df.columns = self.ds_names

        self.metadata["processed_rows"] = len(final_df)
        self.metadata['processed_stats'] = final_df.describe()
        final_df.to_csv(fpath / fname)
        self.metadata["processed_file"] = fpath / fname
        self.metadata["processed_file_stats"] = (fpath / fname).stat()
        del final_df

    @property
    def final_df(self):
        if "processed_file" not in self.metadata:
            raise RuntimeError("Cannot access final product. Data is still being processed.")

        if not hasattr(self, '_final_df'):
            resp = pd.read_csv(self.metadata['processed_file'])
            self._final_df = resp

        return self._final_df

    async def collect_and_write(self, table, select_iter):

        fpath = Path(self.config["DEFAULT"]["bigfile_tmp_path"]) / self.jobid
        fpath.mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.utcnow()
        fname = f"{table}_{now.strftime('%m%d%H%M')}.csv"

        self.metadata['tables'][table]["tmpfile"] = str(fpath / fname)

        is_first = True

        async for records in select_iter:

            dataframe = pd.DataFrame.from_records(
                records,
                columns=("timestamp", "value"),
                index="timestamp")
            self.metadata['tables'][table]['rows_written'] += len(dataframe)
            dataframe.to_csv(fpath / fname, header=is_first, mode='a')
            if is_first:  # only write header on first iteration
                is_first = False

        del dataframe

        self.metadata['tables'][table]['finished'] = True
        return fpath / fname

    def __repr__(self):

        return str(self.state)

    def __str__(self):
        return str(self.state)



    def __getitem__(self, item):
        return self.metadata[item]

    def __getattr__(self, attr):
        return self.metadata[attr]
    @property
    def json(self):
        return json.dumps(self.state, indent=4, default=str)

    @property
    def state(self):
        resp = self.metadata
        resp["t0_utc"] = self.start
        resp["tf_utc"] = self.stop
        resp["tspan_hours"] = (self.stop - self.start).total_seconds() / 3600
        resp['fit_order'] = self.fit_order

        return resp

    @property
    def sample_times(self):
        return pd.date_range(self.start, self.stop, periods=self.nsamples)

    @property
    def tables(self):
        return self.metadata['tables']

    def down_sample(self, csv_file, table):
        """Read csv_file in chunks using an evenly spaced sample time
        to downsample"""
        chunks = pd.read_csv(csv_file, chunksize=10000)
        sample_times = self.sample_times
        out_df = pd.DataFrame()

        for ii, df in enumerate(chunks):
            df.index = pd.to_datetime(df.timestamp * 1000000)  # Pandas dt is in nanoseconds
            del df["timestamp"]
            subsample = sample_times[sample_times < df.index.max()]
            subsample = subsample[subsample > df.index.min()]
            with_sample_times = pd.concat([df, pd.DataFrame([np.nan] * len(subsample), index=subsample)])
            newdf = with_sample_times.value.interpolate(method='polynomial', order=self.fit_order)[subsample]

            out_df = pd.concat([out_df, newdf])


        return out_df



class job_interface:
    config = AppConfig()
    class _job_interface:
        def __init__(self):
            self._jobs = {}

        def submit_job(self, job):
            task = asyncio.create_task(job.run())
            self._jobs[job.jobid] = {"task": task, "job": job}


        def keys(self):
            return self._jobs.keys()

        def __getitem__(self, uuid):
           return self._jobs[uuid]

        def items(self):
            for key, value in self._jobs.items():
                yield key, value
            for data in Path(self.config["DEFAULT"]["bigfile_path"]).iterdir():

                yield






    instance = None

    def __new__(cls):

        if cls.instance is None:
            cls.instance = cls._job_interface()

        return cls.instance
