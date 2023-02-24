import pandas as pd
import numpy as np
import random
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

    # def _set_cuts_df(self, initial = False):
    #     if initial:
    #           df = self.prod[['lg', 'rep']]
    #           df['rest'] = self.params["load"]["stock_lenght"] - df.lg
    #           df.index = [f"_{x}" for x in df.rep]
    #           df = df.drop(columns = ['lg', 'rep'])
    #           self.av_cuts = df
        
    #     return df
    
    # def _make_cuts(self, reps):
        # reps = reps.split("_")[1:]
        # df = self.prod.loc[self.prod.rep.isin([int(x) for x in reps])].copy()
        # if self.cuts is None:
        #     df['count'] = 1 
        # else:
        #     df['count'] = self.cuts["count"].iloc[-1] + 1
        # df["depth"] = len(reps)
        # df["rest"] = np.NaN
        # df.loc[df.index[-1], "rest"] = self.params["load"]["stock_lenght"] - np.sum(df.lg)
        # self.cuts = pd.concat([self.cuts, df])
        # self._set_prod(self.prod.loc[~self.prod.rep.isin(list(self.cuts.rep))])
        
    # def _cut_unsalvageable(self):
    #     cuts = self.av_cuts.loc[self.av_cuts.rest < self.params["storage"][0][0]]
    #     for n in cuts.index:
    #         self._make_cuts(n)
    #     self.av_cuts = self._set_cuts_df(initial=True)
  
    # def _update_av_cuts(self):
    #     pass

    # def add_depth(self):
    #     df = self.av_cuts.copy()
    #     dive = False
    #     depth = 1
    #     for rep, lg in zip(self.prod.rep, self.prod.lg):
    #         if lg <= np.max(df.rest):
    #             df['reps'] = [y for y in[x.split("_")[1:] for x in df.index]]
    #             df[str(rep)] = df.apply(lambda row : np.NaN if str(rep) in row['reps'] else row['rest'] - lg, axis=1)
    #             depth = len(df.loc[df.index[-1], "reps"])
    #             df = df.drop(columns = ['reps'])
    #     df[df <= 0] = np.NaN
    #     ic(df)
    #     vcuts = []
    #     for col in df.columns[1:]:
    #         vcuts.extend([f"{x}_{col}" for x in df[col].dropna().index if int(col)< int(x.split("_")[-1])])
    #     idx = df.index.tolist() + vcuts
    #     ic(idx)
    #     cdf = pd.DataFrame(index = idx)
    #     return cdf
    
    # def _set_rest_col(self, idx) : 
    #     """makes a 'rest' column from the dataframe index"""
    #     ic(self.prod, self.prod.dtypes)
    #     df = idx
    #     ls = [y.split("_")[1:] for y in [x for x in df.index]]
    #     rest = []
    #     for n in ls : 
    #         _ls = [int(x) for x in n]
    #         s = self.params['load']['stock_lenght'] - np.sum(self.prod.loc[self.prod.rep.isin(_ls), 'lg'])
    #         rest.append(s
    #                     )
    #     df['rest'] = rest
    #     self.av_cuts = df



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
            cut = self.prod.loc[self.prod.rep.isin([int(x) for x in (reps.split("_"))])].copy()
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
        if isinstance(df.loc[df.index[0], 'reps'], np.int64):
            df['iscut'] = df.reps.isin([int(x) for x in self.cuts.rep])
        self.avlbl_cuts = df.loc[~df.iscut].drop(columns=['iscut'])
        return df.loc[~df.iscut].drop(columns=['iscut'])

    def cut_forced_waste(self):
        """adds to cuts dataframe items that cannot be concatenated in a stock lenght
        """
        df = self.avlbl_cuts.copy()
        df = df.loc[df.rest<self.params["storage"][0][0]]
        for rep in df.reps:
            self._make_cut(str(rep))

    def add_depth(self):

        def get_rest_col(df):
            # reps = [int(x) for x in y.split("_")[1:] for y in df.re)]
            df['rest'] = [self.params['load']["stock_lenght"] - self.prod.loc[self.prod.rep.isin([int(y) for y in x.split("_")]), 'lg'] for x in df.reps]
            return df

        df = self.avlbl_cuts.copy()
        if np.issubdtype(df.reps, np.int64):
            df['reps'] = [f"_{x}" for x in df['reps']]
        for rep, lg in zip(self.prod.rep, self.prod.lg):
            if lg <= np.max(df.rest):
                df[str(rep)] = df.apply(lambda row : 
                                        np.NaN if str(rep) in row['reps'].split("_") 
                                        else row['rest'] - lg, axis=1)
        df.index = df.reps
        df = df.drop(columns=['rest', 'reps'])
        df[df<0] = np.NaN
        df = df.dropna(how = 'all', axis=0).dropna(how = 'all', axis=1)
        ic(df)
        vcuts = []
        for col in df.columns[1:]:
            vcuts.extend([f"{x}_{col}" for x in df[col].dropna().index if int(col)< int(x.split("_")[-1])])
            if len(vcuts) == 0:
                return None
        df = pd.DataFrame({"reps":df.index.tolist() + vcuts})
        df = get_rest_col(df)
        ic(df)

    

sim = Simulator()    
s = Hdcs()

prod = sim.simulate_tube60_prod(10)
s.init_prod(prod)
s._init_cuts_df()
s.cut_forced_waste()
s.add_depth()







# s._set_cuts_df(True)
# s._cut_unsalvageable()
# df = s.add_depth()
# s._set_rest_col(df)





