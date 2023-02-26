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
        self.cuts = None
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


    def _init_cuts_df(self):
        """sets the 'rest' col on avlbl_cuts based on reps column
        """
        df = pd.DataFrame(columns=['reps', 'rest'])
        df['reps'] = self.prod.rep
        df['rest'] = self.params['load']['stock_lenght'] - self.prod.loc[self.prod.rep.isin(list(df.reps)), 'lg']
        self.avlbl_cuts = df
    
    def _make_cut(self, reps, origin="new", excess="waste"):   
        count = 1 if self.cuts is None else self.cuts['count'].iloc[-1] +1
        if not "_" in reps:
            cut = self.prod.loc[self.prod.rep == int(reps)].copy()
        else:
            cut = self.prod.loc[self.prod.rep.isin([int(x) for x in reps.split("_")[1:]])].copy()
        cut['count'] = count
        cut.loc[cut.index[-1], 'rest'] = self.params['load']['stock_lenght']-np.sum(cut.lg)
        cut.loc[cut.index[0], 'origin'] = origin
        cut.loc[cut.index[-1], 'excess'] = excess
        cut.loc[:,'depth'] = len(reps) if isinstance(reps, list) else 1
        if self.cuts is None:
            self.cuts = cut
        else:
            self.cuts = pd.concat([self.cuts, cut])
        self._update_prod()
        self._update_avlbl_cuts()
    
    def _update_prod(self):
        self.prod = self.prod.loc[~self.prod.rep.isin(self.cuts.rep)]

    def _update_avlbl_cuts(self):
        if self.avlbl_cuts is None:
            return None
        df = self.avlbl_cuts.copy()
        ic(df)
        ic(self.cuts)
        if isinstance(df.loc[df.index[0], 'reps'], np.int64):
            df['reps'] = [f"_{x}" for x in df.reps]
        df["iscut"] = df.apply(lambda row: True if any([z == y] for z in row["reps"].split("_")[1:] for y in [str(x) for x in self.cuts.rep]) else False, axis = 1)
        ic(df)
        self.avlbl_cuts = df.loc[~df.iscut].drop(columns=['iscut'])
        return df.loc[~df.iscut].drop(columns=['iscut'])

    def cut_forced_waste(self):
        """adds to cuts dataframe items that cannot be concatenated in a stock lenght
        """
        df = self.avlbl_cuts.copy()
        df = df.loc[df.rest<self.params["storage"][0][0]]
        for rep in df.reps:
            self._make_cut(str(rep))

    def _add_depth(self):

             
        df = self.avlbl_cuts.copy()
        if np.issubdtype(df.reps, np.int64):
            df['reps'] = [f"_{x}" for x in df['reps']] #for the first iteration
        df = self._get_rest_col(df)
        depth = np.max([len(x.split("_")) for x in df.reps])-1
        st = time.time()
        df = self._add_combination(df)
        ic(f"comb time : {time.time() - st}")
        if df is None:
            return None
        valids = self._locate_positive_values(df)
        return valids

    def set_max_depths(self):
        while True :
            valid_cuts = self._add_depth()
            if valid_cuts is None:
                break
            else: 
                ic('Adding another depth')
                df = self.avlbl_cuts.copy()
                if np.issubdtype(df.reps, np.int64):
                    df['reps'] = [f"_{x}" for x in df['reps']] #for the first iteration
                df = pd.DataFrame({"reps" : df.reps.tolist() + valid_cuts})
                self.avlbl_cuts = self._get_rest_col(df)
                # ic(self.avlbl_cuts)

    def cut_best_fits(self):
        df = self.avlbl_cuts.copy()
        df['depth'] = [len(x.split("_")[1:]) for x in df.reps]
        df = df.sort_values(by=['depth', 'rest'], ascending=[False, True])

        fitdf = pd.DataFrame()
        maxi = self.params['rules']["acceptable"]

        while True :
            fitdf = self.avlbl_cuts.loc[self.avlbl_cuts.rest<maxi].sort_values(by = "rest").copy()
            if len(fitdf) == 0:
                break
            self._make_cut(fitdf.reps.iloc[0])
            ic(self.cuts, self.prod, self.avlbl_cuts)




    """DATAFRAME MANIUPLATION METHODS"""

    def _get_rest_col(self, df):
        """Will add a "rest" column
        REQUIRED "reps" column

        Args:
            df (pd.DataFrame, optional): Dataframe to process
        """
        df['rest'] = [self.params['load']["stock_lenght"] - np.sum(self.prod.
                                                                loc[self.prod.rep.isin(
                                                                [int(y) for y in x.split("_")[1:]]), 'lg']) 
                                                                for x in df.reps]
        return df
    
    def _add_combination(self, df):
        """Tries to sum what's lef in self.prod to each previous combination. Reps in index for further transfomrations

        Args:
            df (pd.DataFrame, optional): REQUIRED : "reps". Defaults to None.

        Returns:
            _type_: _description_
        """

        df['depth'] = [len(x.split("_")[1:]) for x in df.reps]
        # depth = np.max([len(x.split("_")[1:]) for x in df.reps])
        df = df.loc[df.depth == np.max(df.depth)]
        df = df.drop(columns = ['depth'])
        for rep in self.prod.rep:
            df[str(rep)]  = df.apply(lambda row : np.NaN 
                                    if (str(rep) in row['reps'].split("_")) 
                                    else row['rest'] - self.prod.loc[self.prod.rep == rep, "lg"].iloc[0], axis =1)
        df.index = df.reps
        df = df.drop(columns=['reps', 'rest'])
        df[df<0] = np.NaN
        df = df.dropna(how = "all", axis = 0).dropna(how = "all", axis = 1)
        
        return None if len(df) == 0 else df

    def _locate_positive_values(self, df):
        """locates positive values in df 
        Returns a list of tuples with coords : [("row", "col")]"""
        # ic(df)
        valids = pd.DataFrame(np.where(df>=0)).transpose()
        valids.columns = ['row', 'col']
        depth = np.max([len(x.split("_")) for x in df.index])-1
        if depth == 1:
            valids = valids.loc[valids.row<valids.col]
        valids.row = df.index[valids.row]
        valids.col = df.columns[valids.col]
        valid_cuts = ["_".join(coords) for coords in list(zip(valids.row.tolist(), valids.col.tolist()))]
        valid_cuts = ["_"+"_".join(sorted(x.split("_")[1:])) for x in valid_cuts]

        valid_cuts = list(dict.fromkeys(valid_cuts))

        if len(valid_cuts) == 0:
            return None
        return valid_cuts
        








sim = Simulator()    
s = Hdcs()

prod = sim.simulate_tube60_prod(30)
# ic(prod)
s.init_prod(prod)
ic(s.prod)
s._init_cuts_df()
# s.cut_forced_waste()
ic(s.prod)
s.set_max_depths()
df = s.avlbl_cuts
df['depth'] = [len(x.split("_")[1:]) for x in df.reps]
ic(f"{len(df)} possible combinations")
ic(df.sort_values(by=['depth', 'rest'], ascending=[False, True]))
s.cut_best_fits()








# s._set_cuts_df(True)
# s._cut_unsalvageable()
# df = s.add_depth()
# s._set_rest_col(df)





