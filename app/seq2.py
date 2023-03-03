import pandas as pd
import numpy as np
import time
import json
from sequencer import Simulator
from icecream import ic

ic.configureOutput(includeContext=True)

class Hdcs:
    def __init__(self, path_to_config_file = "app/cfg.json"):
        self.cfg_path = path_to_config_file
        self.params = self._init_from_json()
        self.storage = self._init_storage()
        self.prod = None
        self.cuts = pd.DataFrame()
        self.avlbl_cuts = None

    """INIT METHODS"""
    
    def _init_from_json(self):
        with open(self.cfg_path, "r") as f:
            params = json.load(f)
        return params
    
    def _init_storage(self):

        df = pd.DataFrame({"mini": [n for n in range(self.params['storage']["min"], 
                                                  self.params['storage']["max"], 
                                                  self.params['storage']["step"])]})
        df["maxi"] = df['mini'].shift(-1)
        
        df = df.dropna()
        df["name"] = [str(x/1000).replace(".","m") for x in df.mini]
        df['qtt'] = 0
        # df['split'] = [np.mean([x, y]) for x, y in list(zip(df.mini, df.maxi))]
        self.storage = df

    
    def _set_prod(self, prod):
        self.prod = prod

    def _deploy_quantities(self):

        df = self.prod.copy()
        multiples = pd.DataFrame.from_records(
            [x for x in df.loc[df.qtt>1]
             .to_dict('records') for n in range(x['qtt'])])
        multiples['qtt'] = 1
        df = pd.concat([df.loc[df.qtt  == 1], multiples], ignore_index=True)
        df = df.sort_values(by="lg").reset_index(drop=True)
        df["rep"] = df.index
        df = df.drop("qtt", axis=1)
        self._set_prod(df)

    def init_prod(self, prod):
        self._set_prod(prod)
        self._deploy_quantities()
        ic(f"TOTAL : {len(self.prod)} items")

    """MAIN METHODS"""

    def make_cut(self, reps, origin = "new", excess = "waste"):

        cnt = 1 if len(self.cuts)== 0 else self.cuts.cnt.iloc[-1]+1
        ints = self._get_int_list_from_reps(reps)
        cut = self.prod.loc[self.prod.rep.isin(ints)].copy()
        cut['depth'] = self._get_depth_from_reps(reps)
        cut.loc[cut.index[0], "origin"] = origin
        cut.loc[cut.index[-1], "excess"] = excess
        cut.loc[cut.index[-1], "rest"] = self.params['load']["stock_lenght"] - np.sum(cut.lg)
        cut.loc[:, "cnt"] = cnt

        if "s_" in excess:
            name = excess.split("_")[-1]
            self.storage.loc[self.storage.name == name, "qtt"] += 1
        if "s_" in origin:
            name = origin.split("_")[-1]
            self.storage.loc[self.storage.name == name, "qtt"] -= 1
            cut.loc[cut.index[-1], "rest"] = self.params['storage']["step"]/2 #half a step of waste
        self.cuts = pd.concat([self.cuts, cut])
        self.prod = self.prod.loc[~self.prod.rep.isin(ints)]
        self.update_avlbl_cuts()
        self.update_storage_status()
        # ic(self.storage)

    def cut_forced_waste(self):
        df = self.avlbl_cuts.copy()
        df = df.loc[df.rest<self.params["storage"]["min"]]
        for rep in df.index.tolist():
            self.make_cut(rep)

    def cut_best_fits(self, fit):
        while True:
            df = self.avlbl_cuts.copy()
            df = df.loc[df.rest <= fit]
            if df.empty:
                return None
            df = df.sort_values(by = ["rest", "depth"], ascending = [True, False])
            self.make_cut(df.index[0])

    def run_sequencer(self, prod):
        # ic(prod)
        self.init_prod(prod)
        self.init_cuts_df()
        self.cut_forced_waste()
        self.get_max_depth_cuts()
        ic(f"Processing {len(self.avlbl_cuts)} combinations")

        # self.cut_best_fits()
        # ic(self.cuts, self.prod, self.storage)



    """DATAFRAME MANIUPLATION METHODS"""
        
    def init_cuts_df(self):
        df = pd.DataFrame(index = [f"_{x}" for x in self.prod.rep])
        df['rest'] = self.params['load']['stock_lenght'] - df.index.map(self._get_lg_from_reps)
        df['depth'] = df.index.map(self._get_depth_from_reps)
        self.avlbl_cuts = df
        return df
    
    def _get_int_list_from_reps(self, reps):
        """gei the integer list of the reps string

        Args:
            reps (str): _rep_rep etc

        Returns:
            list: [int, int, int etc...]
        """
        ls = [int(x) for x in reps.split("_")[1:]]
        return ls

    def _get_lg_from_reps(self, reps):
        """get lenght of combination

        Args:
            reps (str): _rep_rep etc

        Returns:
            int: lenght from the prod dataframe
        """
    
        return np.sum(self.prod.loc[self.prod.rep.isin(
                                    self._get_int_list_from_reps(reps)), "lg"])

    def _get_depth_from_reps(self, reps):
        return len(self._get_int_list_from_reps(reps))

    def update_avlbl_cuts(self):
        df = self.avlbl_cuts.copy()
        df['iscut'] = [any([x in (self.cuts.rep) for x in z]) for z in [self._get_int_list_from_reps(x) for x in df.index]]
        self.avlbl_cuts = df.loc[~df.iscut].drop(columns="iscut")


    """COMBINATIONS BUILDER"""

    def _add_depth(self):
        """Generates the dataframe of the next depth level

        Returns:
            pd.DataFrame: _description_
        """
        df = self.avlbl_cuts.copy()
        df = df.loc[df.depth == np.max(df.depth.tolist())]
        ls = [pd.Series([x - self.prod.loc[self.prod.rep == y, "lg"].iloc[0]
                          for x in df.rest],name = y) 
                          for y in self.prod.rep 
                          if self.prod.loc[self.prod.rep == y, "lg"].iloc[0] < np.max(df.rest)]
        if len(ls)==0:
            return None
        conc = pd.concat(ls, axis=1)
        conc.index = df.index
        # ic(conc)
        return conc

    def locate_possible_cuts(self, conc_df):
        """get the list of all possible combinations

        Args:
            conc_df (pd.DataFrame): Dataframe from _add_depth

        Returns:
            None|list: None if no cuts possible else list of cuts
        """
        idx = self.avlbl_cuts.index.tolist()
        df = conc_df
        # ic(df)
        valids = pd.DataFrame(np.where(df>0), index = ["row", "col"]).transpose()
        valids.row = df.index[valids.row]
        valids.col = df.columns[valids.col]
        ints = [self._get_int_list_from_reps(x) for x in valids.row]
        valids['dupli'] = [x in y for x, y in zip(valids.col, ints)]
        valids = valids.loc[~valids.dupli]
        valids.col = valids.col.astype(str)

        #make list : 
        valid_cuts = ["_".join(coords) for coords in list(zip(valids.row.tolist(), valids.col.tolist()))]
        valid_cuts = ["_"+"_".join(sorted(x.split("_")[1:])) for x in valid_cuts]
        valid_cuts = list(dict.fromkeys(valid_cuts))
        return idx + valid_cuts if len(valid_cuts)>0 else None
        
    def get_max_depth_cuts(self):
        """generates the list of all the possible cuts

        Returns:
            None: update avlbl_cuts
        """
        while True:
            df = s._add_depth()
            if df is None:
                return None
            idx = s.locate_possible_cuts(df)
            if idx is None:
                return None
            self.avlbl_cuts = pd.DataFrame(index = idx)
            self.avlbl_cuts['rest'] = self.params['load']['stock_lenght'] - self.avlbl_cuts.index.map(self._get_lg_from_reps)
            self.avlbl_cuts['depth'] = self.avlbl_cuts.index.map(self._get_depth_from_reps)
            # ic(len(self.avlbl_cuts))
        


    """STORAGE MANAGEMENT"""

    def update_storage_status(self):
        df = self.storage
        df['status'] = ["CRIT_FILL" if x<self.params['storage']['crit'] 
                                    else ("CRIT_EMPTY" if x>=self.params['storage']['capacity']-self.params['storage']['crit']
                                          else "OPEN") for x in self.storage.qtt]
        self.storage = df

    def _get_store_column(self, retrieve=False):

        df = self.avlbl_cuts
        store = self.storage


        if retrieve:
            step = self.params["storage"]["step"]
            df['lg'] = self.params["load"]["stock_lenght"] - df.rest
            return[x in range(int(np.min(store.mini+step)), int(np.max(store.maxi-step))+1) for x in df.lg]

        return[x in range(int(np.min(store.mini)), int(np.max(store.maxi))+1) for x in df.rest]

    def _get_good_storage_fits(self, fit = None, crit_only = False):
        if fit is None : 
            fit = self.params["storage"]["step"]
        df = self.avlbl_cuts.copy()
        df['storable'] = self._get_store_column()
        df = df.loc[df.storable]
        df.drop(columns=['storable'])
        rgs = [range(x, x+fit) for x in self.storage.mini]
        df['good'] = [any(x in y for y in rgs) for x in df.rest]
        df = df.loc[df['good']]
        df['location'] = self.get_storage_location_column(df)
        ic(df)
        # ic(self.storage.loc[self.storage.status == "CRIT_FILL", "name"].tolist())
        if crit_only : 
            df = df.loc[df.location.isin("s_" + str(x) for x in [self.storage.loc[self.storage.status == "CRIT_FILL", "name"].tolist()])]
        else :
            df = df.loc[df.location.isin("s_" + str(x) for x in [self.storage.loc[self.storage.status != "CRIT_EMPTY", "name"].tolist()])]
        ic(df)
        df = df.sort_values(by ='depth', ascending=False)
        return list(zip(df.index.tolist(), df.location.tolist()))
    
    def _get_good_retrieve_fits(self, fit = None, crit_only = False):
        if fit is None : 
            fit = self.params["storage"]["step"]
        df = self.avlbl_cuts.copy()
        df['lg'] = self.params["load"]["stock_lenght"] - df.rest
        df['storable'] = self._get_store_column(True)
        # ic(df)
        df = df.loc[df.storable]
        # ic(df)
        df.drop(columns=['storable'])
        rgs = [range(x-fit, x) for x in self.storage.mini]
        # ic(rgs)
        df['good'] = [any(x in y for y in rgs) for x in df.lg]
        df = df.loc[df['good']]
        df['location'] = self.get_storage_location_column(df, True)
        ic(self.storage)
        if crit_only : 
            df = df.loc[df.location.isin(self.storage.loc[self.storage.status == "CRIT_EMPTY", "name"].tolist())]
        else :
            df = df.loc[df.location.isin(self.storage.loc[self.storage.status != "CRIT_FILL", "name"].tolist())]
        df = df.sort_values(by ='depth', ascending=False)
        # ic(df)
        return list(zip(df.index.tolist(), df.location.tolist()))

    def get_storage_location_column(self, df = None, retrieve = False):
        if df is None:
            df = self.avlbl_cuts.copy()
            df['storable'] = self._get_store_column()
            df = df.loc[df.storable]
            df.drop(columns=['storable'])

        ###TODO INSPECT HERE SOMETHING MESS UP MAPPING

        # rgs = [range(x, y) for x,y in list(zip(self.storage.mini.astype(int), self.storage.maxi.astype(int)))]
        # ic(rgs)
        # d ={k:v for k, v in zip(rgs, self.storage.name)}
        # ic(d)
        # return [v for k,v in d.items() for x in df.rest if x in k]

        r = lambda x : 500 * np.ceil(x/500) if retrieve else 500 * np.floor(x/500)
        return ["s_" + str(r(x)/1000)[:3].replace(".","m") for x in df.rest] if not retrieve else ["s_" + str(r(x)/1000)[:3].replace(".","m") for x in df.lg]

    def cut_good_storage_fits(self, store_fit = 500, retrieve_fit = 500):
        ls = list(self._get_good_storage_fits(store_fit))
        while len(ls) > 0:
            ls = self._get_good_storage_fits(store_fit)
            # ic(ls)
            self.make_cut(ls[0][0], excess=ls[0][1])
            ls = self._get_good_storage_fits(store_fit)

        ls = list(self._get_good_retrieve_fits(retrieve_fit))
        while len(ls) > 0:
            ls = self._get_good_retrieve_fits(retrieve_fit)
            # ic(ls)
            self.make_cut(ls[0][0], origin=ls[0][1])
            ls = self._get_good_retrieve_fits(retrieve_fit)
        
    







sim = Simulator()    
s = Hdcs()
s._init_storage()
ls = []
for n in range(5):
    prod = sim.simulate_tube60_prod(20)
    s.run_sequencer(prod)
    ic(s.prod)
    s.cut_good_storage_fits(100,100)
    s.cut_best_fits(100)
    ic(s.cuts)
    s.cut_good_storage_fits(400, 400)
    s.cut_best_fits(500)
    ic(s.cuts)
    ic(s.storage)
    s.avlbl_cuts = None
    s.cuts = pd.DataFrame()
    s.prod = None
    # ic(s.prod)



    
    
# df = pd.concat(ls).reset_index(drop=True)
# df = df.dropna(subset=['origin', 'excess'])
# ic(df)









# s._set_cuts_df(True)
# s._cut_unsalvageable()
# df = s.add_depth()
# s._set_rest_col(df)





