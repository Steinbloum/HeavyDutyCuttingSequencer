import pandas as pd
import numpy as np
from storage import Storage #COMMENT FOR TESTS
import json
from icecream import ic
import itertools

ic.configureOutput(includeContext=True)

class Hdcs:
    def __init__(self, path_to_config_file = "app/cfg.json"):
        self.cfg_path = path_to_config_file
        ###COMMENT FOR TESTS
        self.params = self._init_from_json()
        self.storage = Storage(path_to_config_file)
        self.storage.init_storage()
        # self.params = None
        ###END COMMENTS
        self.prod = None
        self.cuts = pd.DataFrame()
        self.avlbl_cuts = None

    """INIT METHODS"""
    
    def _init_from_json(self):
        with open(self.cfg_path, "r") as f:
            params = json.load(f)
        ic(params)
        return params
    
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

    def make_cut(self, reps, origin = "new", excess = "waste", update = True):

        cnt = 1 if len(self.cuts)== 0 else self.cuts.cnt.iloc[-1]+1
        ints = self._get_int_list_from_reps(reps)

        cut = self.prod.loc[self.prod.rep.isin(ints)].copy()
        cut['depth'] = self._get_depth_from_reps(reps)
        cut.loc[cut.index[0], "origin"] = origin
        cut.loc[cut.index[-1], "excess"] = excess
        cut.loc[cut.index[-1], "rest"] = self.params['load']["stock_lenght"] - np.sum(cut.lg)
        cut.loc[:, "cnt"] = cnt

        if origin != "new":
            self.storage.retrieve(origin.split("_")[-1])
            cut.loc[cut.index[-1], "rest"] = self.params['storage']["step"]/2
        if excess != "waste":
            self.storage.store(excess.split("_")[-1])
            cut.loc[cut.index[-1], "rest"] = np.NaN
        
        self.cuts = pd.concat([self.cuts, cut])  
        self.prod = self.prod.loc[~self.prod.rep.isin(ints)]
        if isinstance(update, pd.DataFrame):
            df = self.update_avlbl_cuts(update)
        else :
            self.update_avlbl_cuts()
            ic("CUT DONE", cut)
            return True
        
        ic("CUT DONE", cut)
        return df
        
    def cut_forced_waste(self):
        df = self.avlbl_cuts.loc[self.avlbl_cuts.depth == 1].copy()
        df = df.loc[df.rest < np.min(df.lg)]
        for rep in df.index:
            df = self.make_cut(rep, update= df)

    def _retrieve_critical(self):
        # ic(self.avlbl_cuts)
        df = self.avlbl_cuts.loc[self.avlbl_cuts.rtv_status.isin(['FULL', "CRIT_FULL"])].copy()

        ic(df)

        while True:
            df = self.avlbl_cuts.loc[self.avlbl_cuts.rtv_status.isin(['FULL', "CRIT_FULL"])].copy()
            if len(df) == 0:
                break
            df["delta"] = [self.storage.storage.loc[self.storage.storage.name == x, "mini"].iloc[0]
                       for x in df.rtv_loc]-df.lg
            df = df.sort_values(by=["depth","delta"], ascending=[True, True])
            ic(df)
            self.make_cut(df.index[0], df.rtv_loc[0])

    def _retrieve_max(self, inpt=None):
        """cut max from storage

        Args:
            inpt (None, pd.DataFrame, optional): if custom avalbable_df. Defaults to None.
        """        

        df = inpt if inpt is not None else self.avlbl_cuts
        df = df.loc[~(df.rtv_status == "EMPTY")
                                        ].sort_values(by="depth",ascending=False
                                        ).dropna(subset="rtv_loc"
                                        ).copy()
        ic(df)
        while len(df) > 0:
            ic(df)
            df = self.make_cut(df.index[0], origin=df.rtv_loc[0], update=df)
        if inpt is not None:
            return self.update_avlbl_cuts(inpt)
        self.update_avlbl_cuts()

    def _store_max(self, inpt=None):
        """store max from storage

        Args:
            inpt (None, pd.DataFrame, optional): if custom avalbable_df. Defaults to None.
        """        

        df = inpt if inpt is not None else self.avlbl_cuts
        df = df.loc[~(df.store_status == "FULL")].dropna(subset="store_loc").copy()
        ic(df)
        while len(df) > 0:
            ic(df)
            df = self.make_cut(df.index[0], excess=df.store_loc[0], update=df)
            df = df.loc[~(df.store_status == "FULL")].dropna(subset="store_loc")
        if inpt is not None:
            return self.update_avlbl_cuts(inpt)
        self.update_avlbl_cuts()

    def _store_critical(self):

        df = self.avlbl_cuts.loc[self.avlbl_cuts.store_status.isin(['EMPTY', "CRIT_EMPTY"])].copy()

        ic(df)

        while True:
            df = self.avlbl_cuts.loc[self.avlbl_cuts.store_status.isin(['EMPTY', "CRIT_EMPTY"])].copy()
            if len(df) == 0:
                break
            df["delta"] = df.rest-[self.storage.storage.loc[self.storage.storage.name == x, "mini"].iloc[0]
                       for x in df.store_loc]
            df = df.sort_values(by=["delta","depth"], ascending=[True, False])
            ic(df)
            self.make_cut(df.index[0], excess = df.store_loc[0])

    def _cut_good_fits(self, fit = "acceptable", avlbl = False):
        if isinstance(fit, str):
            fit = self.params['rules'][fit]
        df = avlbl if isinstance(avlbl, pd.DataFrame) else self.avlbl_cuts.copy()
        df = self.avlbl_cuts.loc[self.avlbl_cuts.rest <= fit].sort_values(by = ['rest', 'depth'], 
                                                                            ascending = [True, False])
        
        while True:
            df = self.avlbl_cuts.loc[self.avlbl_cuts.rest <= fit].sort_values(by = ['rest', 'depth'])
            if len(df) == 0:
                break
            ic(df)
            self.make_cut(df.index[0])

    def run_sequencer(self, prod):
        # self._init_from_json()
        self.init_prod(prod)
        self.init_cuts_df()
        self.get_max_depth_combinations()
        self.cut_forced_waste()
        self._retrieve_critical()
        self._store_critical()
        self._cut_good_fits()
        self._cut_good_fits(500)



    """DATAFRAME MANIUPLATION METHODS"""
        
    def init_cuts_df(self):
        df = pd.DataFrame(index = [f"_{x}" for x in self.prod.rep])
        df['lg'] = df.index.map(self._get_lg_from_reps)
        df["rest"] = self.params['load']['stock_lenght']-df.lg
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

    def update_avlbl_cuts(self, inpt = False):
        df = self.avlbl_cuts.copy() if not isinstance(inpt, pd.DataFrame) else inpt
        df['iscut'] = [any([x in (self.cuts.rep) for x in z])
                        for z in [self._get_int_list_from_reps(x) 
                        for x in df.index]]
        if not isinstance(inpt, pd.DataFrame):
            self.avlbl_cuts = df.loc[~df.iscut].drop(columns="iscut")
            self.avlbl_cuts = self.get_storage_info()
        else:
            df = df.loc[~df.iscut].drop(columns="iscut")
            return self.get_storage_info(df)
        
    def _get_priority_cuts(self):
        """returns the dataframe of the combinations containing reps we can not store

        Returns:
            pd.Dataframe: Priority cuts dqtaframe
        """   
        def compare(tupl):
            t = tupl
            cnt = 0
            for rep in t[0]:
                if rep in t[1]:
                    cnt +=1
            return cnt
        
                
        ls = self._get_unstorable_list()
        df = self.avlbl_cuts.copy()
        # df = df.loc[df.depth == 2]
        ic(df.index)
        # ic(ls)
        ints = [self._get_int_list_from_reps(x) for x in df.index]
        # ic(len(ints))
        ts = [compare((x, ls)) for x in ints]
        df['prio'] = ts
        # nb = [cnt(((x, ls)))]
        df = df.loc[df.prio > 0]
        df = df.sort_values(by=['rest','prio', 'depth'], ascending=[True, False, False])
        return df

    """COMBINATIONS BUILDER"""

    def _add_depth(self):
        """Gets the dataframe of the next combination level

        Returns:
            pd.DataFrame: index = _rep_rep, cols = summed reps
        """        
        df = self.avlbl_cuts.copy()
        df = df.loc[df.depth == np.max(df.depth.tolist())]
        # ic(df)
        max_val = np.max(df.rest)
        d = {rep:[x-lg for x in df.rest] 
            for rep, lg 
            in list(zip(self.prod.rep,self.prod.lg)) 
            if lg<max_val}

        conc = pd.DataFrame.from_dict(d)
        conc.index = df.index
        return conc

    def _get_valid_combinations_dataframe(self, df):
        """gets the df of all unique possible combinations

        Args:
            df (pd.Dataframe): df from self.add_depth

        Returns:
            pd.Dataframe: to add to index
        """
        valids = pd.DataFrame(np.where(df>0), index = ["row", "col"]).transpose()
        valids.row = df.index[valids.row]
        valids.col = df.columns[valids.col]
        ints = [self._get_int_list_from_reps(x) for x in valids.row]
        valids['dupli'] = [x in y for x, y in zip(valids.col, ints)]
        valids = valids.loc[~valids.dupli]
        valids.col = valids.col.astype(str)
        valid_cuts = ["_".join(coords) for coords in list(zip(valids.row.tolist(), valids.col.tolist()))]
        valid_cuts = ["_"+"_".join(sorted(x.split("_")[1:])) for x in valid_cuts]
        valid_cuts = list(dict.fromkeys(valid_cuts))
        df = pd.DataFrame(index=valid_cuts)
        df['lg'] = df.index.map(self._get_lg_from_reps)
        df["rest"] = self.params['load']['stock_lenght']-df.lg
        df['depth'] = df.index.map(self._get_depth_from_reps)
        if len(valid_cuts)>0:
            ic(f"Depth {df.depth.iloc[-1]} : {len(df)} combinations")

        return df if len(valid_cuts)>0 else None

    def get_max_depth_combinations(self, storage = True):

        df = self.avlbl_cuts.copy()
        while True:
            df = self._get_valid_combinations_dataframe(self._add_depth())
            if df is None:
                break
            self.avlbl_cuts = pd.concat([self.avlbl_cuts,df]) 
        if storage:      
            self.avlbl_cuts = self.get_storage_info()
        ic(f"TOTAL COMBINATIONS : {len(self.avlbl_cuts)}")
        return self.avlbl_cuts

    def _get_unstorable_list(self):
        df = self.avlbl_cuts.loc[self.avlbl_cuts.depth == 1].copy()
        df = df.loc[~df.index.isin(df.dropna(subset=["store_loc"],axis=0).index)]
        ls = list(itertools.chain.from_iterable([self._get_int_list_from_reps(x)for x in df.index]))
        return ls

    """STORAGE MANAGEMENT"""

    def get_storage_info(self, inpt = False):
        
        avlbl = self.avlbl_cuts if not isinstance(inpt, pd.DataFrame) else inpt
        avlbl['store_loc'] = [self.storage.get_unique_location(x) for x in avlbl.rest]
        avlbl["store_status"] = [self.storage.get_status(x) for x in avlbl.store_loc]
        avlbl['rtv_loc'] = [self.storage.get_unique_location(x, "retrieve") for x in avlbl.lg]
        avlbl["rtv_status"] = [self.storage.get_status(x) for x in avlbl.rtv_loc]

        return avlbl







