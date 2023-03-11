import pandas as pd
import numpy as np
import json
from icecream import ic

ic.configureOutput(includeContext=True)


class Storage:

    def __init__(self, cfg_path) -> None:
        self.storage = None
        with open(cfg_path, "r") as f:
            self.cfg = json.load(f)['storage']
        self.init_storage()
        self.loc_mapping_dict = self._get_location_map_dict()

    def init_storage(self):
        """initialise the storage dataframe from the json cfg file
        """
        cfg = self.cfg
        df = pd.DataFrame({"mini":[x for x in range(cfg['min'], cfg['max'] + 1, cfg['step'])]})
        df['maxi'] = df.mini.shift(-1)
        df = df.dropna().astype(int)
        df['qtt'] = 0 
        df['status'] = None
        df['name'] = [str(x/1000).replace(".", "m") for x in df.mini]
        self.storage = df
        self._update_status()

    def _update_status(self):
        """updates status column based on qtt values"""
        df = self.storage
        crit = {"min" : int(self.cfg["capacity"] * (self.cfg['crit']/100)),
                "max" : int(self.cfg["capacity"]- (self.cfg["capacity"] * (self.cfg['crit']/100)))}
        df['status'] = ["CRIT_FILL" if x<crit['min'] 
                                    else ("CRIT_EMPTY" if x>=crit['max']
                                          else "OPEN") for x in self.storage.qtt]

    def _get_location_map_dict(self):
        df = self.storage
        d = {}
        d["store"] = {v:k for k, v in
                zip([range(x, y) for x, y in list(zip(df.mini, df.maxi))], df.name)}
        d["retrieve"] = {v:k for k, v in
                zip([range(x-self.cfg['step'], y-self.cfg['step']) 
                     for x, y in list(zip(df.mini, df.maxi))], df.name)}
        return d
    
    def get_unique_location(self, lg, side = "store"):
        """gets the location of the input lenght

        Args:
            lg (int): the lenght to store
            side (str, optional): valid args : "store", "retrieve". Defaults to "store".
                returns the location for the storage or the retrieving

        Returns:
            str, np.NaN: storage location or NaN
        """        
        d = self.loc_mapping_dict[side]
        for k, v in d.items():
            if lg in v:
                return k
        return np.NaN
    
    def get_status(self, name):
        """returns the status of the input storage location. If location does not exists (can be NaN) returns NaN

        Args:
            name (str): name of the location ex : "1m0"

        Returns:
            str, float: status or NaN
        """        
        try:
            return self.storage.loc[self.storage.name == name, "status"].iloc[0]
        except IndexError:
            return np.NaN
    
