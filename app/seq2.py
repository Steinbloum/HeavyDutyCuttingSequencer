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

        return {f"{str(x)[1]}m{str(x)[2:3]}": 
                    {"lg" : range(x[0], x[1]),
                     "qtt" : 0}
                 for x in self.params['storage']}
    
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

        #update variables
        self.cuts = pd.concat([self.cuts, cut])
        self.prod = self.prod.loc[~self.prod.rep.isin(ints)]
        self.update_avlbl_cuts()

    def cut_forced_waste(self):
        df = self.avlbl_cuts.copy()
        df = df.loc[df.rest<self.params["storage"][0][0]]
        for rep in df.index.tolist():
            self.make_cut(rep)

    def cut_best_fits(self):
        while True:
            df = self.avlbl_cuts.copy()
            df = df.loc[df.rest <= self.params["rules"]["acceptable"]]
            if df.empty:
                return None
            df = df.sort_values(by = ["rest", "depth"], ascending = [True, False])
            self.make_cut(df.index[0])

    def run_sequencer(self, prod):
        ic(prod)
        self.init_prod(prod)
        self.init_cuts_df()
        self.cut_forced_waste()
        self.get_max_depth_cuts()
        self.cut_best_fits()


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
            ic(len(self.avlbl_cuts))

sim = Simulator()    
s = Hdcs()

prod = sim.simulate_tube60_prod(50)
# ic(prod)
s.run_sequencer(prod)
ic(s.cuts.fillna(""))
ic(s.avlbl_cuts)
ic(s.prod)









# s._set_cuts_df(True)
# s._cut_unsalvageable()
# df = s.add_depth()
# s._set_rest_col(df)





