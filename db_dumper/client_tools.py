import requests
from configparser import ConfigParser
import pandas as pd
from ipywidgets import widgets, interact
from IPython.display import display
from .appconfig import AppConfig

from abc import ABC, abstractmethod


class widget_container:
    def __init__(self, **wlist):
        interact(self.on_change, **wlist)

    def on_change(self, w, w2):
        print(w, w2)




class db_widget:

    def __init__(self, widget):
        interact(self.on_change, widget=widget)


    def on_change(self, widget):
        print(widget)





class tools:
    def __init__(self):
        self.config=AppConfig()
        self.url = self.config["client"]["json_url"]
        self.info = pd.DataFrame(requests.get(self.config["client"]["info_url"]).json())



    def widgets(self):
        subsystems = list(self.info.subsystem[~self.info.subsystem.duplicated()].values)
        options = [(v,i) for i,v in enumerate(subsystems)]
        subsystems.insert(0, '')
        log = widgets.Dropdown(options=subsystems, descriptions="Log")
        param = widgets.Dropdown(descriptions="Parameter")
        submit = widgets.Button(description='Submit', tooltip='Get Data')



        def on_select(log, params):



            #print(log, param)
            #self.info[self.info.subsystem == log]
            param.options = list(self.info['ds_name'][self.info.subsystem == log])

        def on_submit(value):
            print(value)


        interact(on_select, log=log, params=param)
        display(submit)
        submit.observe(on_submit)






    def junk(self):
        data = {"ds_names": ["laser_cutter_room_temperature3_C", 'hexapod_mini_off_guider_tz_applied'],
                "delta_time": 360000}
        rq = test_it(data=data)

        df = pd.read_json(json.dumps(rq['data']))
        print(rq["errors"])
        print(rq["info"])

    def test_it(self, data=None):
        if data is None:
            data = {"ds_names": ["laser_cutter_room_dewpoint3_C", 'hexapod_mini_off_guider_tz_applied'],
                    "delta_time": 360000}

        url = self.config["client"]["json_url"]
        rq = requests.get(url, json=data)
        try:
            resp = rq.json()
        except Exception as err:
            print(err)
            resp = rq
        return resp
