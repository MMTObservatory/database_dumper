import pandas as pd
import aiomysql
import logging
import datetime
from pathlib import Path
from .appconfig import AppConfig
from uuid import uuid1
import asyncio
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
global_limit = int(AppConfig()["DEFAULT"]["resp_limit"])
import sys
import pickle

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
            resp = await self.query(sqlstr)

            df = pd.DataFrame.from_records(resp, columns=columns)

            return df

        async def long_select(self, table, start, stop):
            """Use fetchmany to yeild the results in chunks.."""

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

    def __init__(self, ds_names: list,
                 nsamples: int = 1000,
                 fit_order: int = 3,
                 startdate = None,
                 enddate = None):
        now = datetime.datetime.utcnow()
        if startdate is None:
            startdate = now-datetime.timedelta(days=6)

        if enddate is None:
            enddate = startdate+datetime.timedelta(days=6)


        self.config = AppConfig()
        self.ds_names = ds_names
        self.stop = enddate
        self.start = startdate
        self.chunksize = 100000
        self.jobid = str(uuid1())
        self.nsamples = nsamples
        self.fit_order = fit_order

        self.metadata = {"job_start_utc": datetime.datetime.utcnow(),
                         'tables': {},
                         'finished': False

                         }

        for table in self.ds_names:
            self.metadata['tables'][table] = {}
            self.metadata['tables'][table]['tmpfile'] = None
            self.metadata['tables'][table]['rows_written'] = 0

    async def run(self):
        """Retrieve raw data from the database, downsample data into evenly
        spaced datapoints and merge the data into one csv_file"""
        logging.debug(f"Beggining run for job {self.jobid}")
        tasks = []
        fpath = Path(self.config["DEFAULT"]["bigfile_path"]) / self.jobid
        fpath.mkdir(exist_ok=True, parents=True)
        fname = "processed.csv"

        # Set up the sql select queries
        for table in self.ds_names:
            logging.debug(f"building select_iter tasks for {table}")
            select_iter = self.conn.long_select(table, self.start, self.stop)
            task = self.collect_and_write(table, select_iter)
            self.metadata['tables'][table]['finished'] = False
            tasks.append(task)


        logging.debug(f"'Gathering' select_iter tasks (querying database and writing raw files)")
        # Get the raw data
        results = await asyncio.gather(*tasks)
        logging.debug("Finished select_iter tasks")
        
        logging.debug("Beginning down sampling")
        # downsample the 
        dfs = [self.down_sample(result, table) for result, table in zip(results, self.ds_names)]
        logging.debug("Finished down sampling")

        final_df = pd.concat(dfs, join='outer', axis=1)
        logging.debug(final_df)
        final_df.columns = self.ds_names

        
        self.metadata["processed_rows"] = len(final_df)
        self.metadata['processed_stats'] = final_df.describe()
        final_df.to_csv(fpath / fname)




        self.metadata["processed_file"] = fpath / fname
        self.metadata["processed_file_stats"] = (fpath / fname).stat()
        self.metadata["plot_file"] = (fpath / "plot.png")

        logging.debug("Beginning plotting")
        try:

            plt.ioff()
            #tmpdf = final_df.dropna()


            ax = final_df.dropna().plot()
            fig = ax.get_figure()
            fig.savefig(self.metadata['plot_file'])

        except Exception as error:
            logging.info(f"There was a plotting error: {error}")

        self.metadata["finished"] = True

        logging.debug(f"Writing metafile {fpath/meta.json}")
        with (fpath/"meta.json").open('w') as fd:
            print(self.metadata)
            json.dump(self.metadata, fd, indent=2, default=str)

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

    def description(self):
        outstr = ""
        for ii, tstring in enumerate(self.ds_names):
            sep = ' | '
            if ii == len(self.ds_names)-1:
                sep = ''

            if len(tstring) > 20:
                outstr += tstring[:17]+'...'+sep
            else:
                outstr += tstring+sep

        return outstr



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
        resp['jobid'] = self.jobid
        resp['description'] = self.description()

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

        logging.debug(f"{table} Chunking csv_file {csv_file}")
        chunks = pd.read_csv(csv_file, chunksize=10000)
        sample_times = self.sample_times
        out_df = pd.DataFrame()

        for ii, df in enumerate(chunks):

            try:
                logging.debug(f"{ii}th chunk of {table} ({len(df)} rows).")
                logging.debug(f"converting to datetime")
                df.index = pd.to_datetime(df.timestamp * 1000000, utc=True)  # Pandas dt is in nanoseconds

                logging.debug(f"{df.index.max()}")
                del df["timestamp"]
                subsample = sample_times[sample_times < df.index.max()]
                subsample = subsample[subsample > df.index.min()]

                with_sample_times = pd.concat([df, pd.DataFrame([np.nan] * len(subsample), index=subsample)])
                logging.debug(f"Subsample times concatenated")
                with_sample_times.to_pickle("wst.pkl")
                df.to_pickle("df.pkl")
                logging.debug(with_sample_times.dtypes)

                newdf = with_sample_times.value.interpolate(method='time')[subsample]
                
                out_df = pd.concat([out_df, newdf])
                logging.debug("New df updated looping...")

            except Exception as error:

                logging.info(f"down_sample error: {error}")


        return out_df


class finished_job:

    def __init__(self, metadata):

        self.state = metadata

    def json(self):

        return json.dumps(self.state, indent=2, default=str)


class job_interface:
    config = AppConfig()
    class _job_interface:
        config = AppConfig()
        def __init__(self):
            self._jobs = {}
            self._tasks = {}

        def submit_job(self, job):
            task = asyncio.create_task(job.run())
            self._jobs[job.jobid] = job
            self._tasks[job.jobid] = task


        def keys(self):
            return self._jobs.keys()

        resp = None
        def __getitem__(self, uuid):
            if uuid in self._jobs:
               resp = self._jobs[uuid]
            else:
                for path in Path(self.config["DEFAULT"]["bigfile_path"]).absolute().iterdir():
                    if str(path.name) == uuid:
                        with (path / Path("meta.json")).open() as fd:
                            meta = json.load(fd)
                            resp = finished_job(meta)
                        break

            if resp is None:
                raise KeyError(f"No such jobid {uuid}")

            return resp


        def iterids(self):
            for jobid in self._jobs.keys():
                yield jobid

            for path in Path(self.config["DEFAULT"]["bigfile_path"]).iterdir():
                yield str(path.name)


        def items(self):
            completed = []
            for jobid, job in self._jobs.items():

                if not job.metadata['finished']:
                    yield jobid, job
                else:
                    completed.append(jobid)

            for cmpid in completed:
                # remove finished jobs
                # This should be done by some sort of
                # cleaning coroutine.
                del self._jobs[cmpid]

            for data in Path(self.config["DEFAULT"]["bigfile_path"]).iterdir():
                if (data/"meta.json").exists():
                    # we should probably do something
                    # in cases were there is no meta.json
                    # file
                    with (data/"meta.json").open() as fd:
                        meta = json.load(fd)

                    yield str(data.name), meta


    instance = None

    def __new__(cls):

        if cls.instance is None:
            cls.instance = cls._job_interface()

        return cls.instance
