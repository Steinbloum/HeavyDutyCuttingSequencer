import pandas as pd
import numpy as np
import json
import random
from icecream import ic


ic.configureOutput(includeContext=True)

class Sequencer:

    def __init__(self, path_to_config_file = "app/cfg.json"):
        self.cfg_path = path_to_config_file
        self.params = self.init_from_json()
        self.storage = self.init_storage()
        self.cuts = []
        self.temp = None

    """INIT METHODS"""
    
    def init_from_json(self):
        with open(self.cfg_path, "r") as f:
            params = json.load(f)
        return params
    
    def init_storage(self):

        return {f"{str(x)[1]}m{str(x)[2:3]}": 
                    {"lg" : range(x[0], x[1]),
                     "qtt" : 0}
                 for x in self.params['storage']}

    """MAIN METHODS"""

    def deploy_quantities(self, production_df):

        df = production_df
        multiples = pd.DataFrame.from_records(
            [x for x in df.loc[df.qtt>1]
             .to_dict('records') for n in range(x['qtt'])])
        multiples['qtt'] = 1
        df = pd.concat([df.loc[df.qtt  == 1], multiples], ignore_index=True)
        df = df.sort_values(by="lg").reset_index(drop=True)
        df["rep"] = df.index
        ic(f"{len(df)} items")
        return df.drop("qtt", axis=1)

    """QUEST FOR BEST FIT METHODS"""

    def build_best_fits_df(self, deployed_df, fit = "perfect"):
        self.temp = {}

        df = deployed_df
        # ic(df)
        cutsdf = self.get_cuts_df(df, True)
        self.add_to_best_fits_recs(cutsdf, fit)
        cutsdf = self.add_depth_level(cutsdf, df)
        n=1
        while cutsdf is not None:
            cutsdf = self.get_cuts_df(df, False, cutsdf)
            self.add_to_best_fits_recs(cutsdf, fit)
            cutsdf = self.add_depth_level(cutsdf, df)
            n+=1
        # ic(self.temp, type(self.temp))
        bestfits = pd.DataFrame.from_dict(self.temp, orient="index", columns=['rest'])
        idx = list(x.split("_")[1:] for x in bestfits.index)
        idx = [[int(x) for x in y] for y in idx]
        bestfits["max_lg"] = [df.loc[df.rep.isin(x), "lg"].max() for x in idx]
        bestfits = bestfits.sort_values(by=["max_lg", "rest"], ascending=[False, True])
        # ic(bestfits)
        return bestfits if len(bestfits)> 0 else None
             
    def get_cuts_df(self, prod_df, first_iter, *args):
        """sums every avalaible lenght to the index combination

        if it is the first iteration, first_iter param must be set to True
        else pass the existing cutdf as argument
        Args:
            prod_df (_type_): _description_
            first_iter (bool, optional): _description_. Defaults to False.
        """
        df = prod_df
        bar_size = self.params['load']['stock_lenght']
        if first_iter:
            cutsdf = pd.DataFrame(index=["_"+str(x) for x in df.rep])
            cutsdf['rest'] = [bar_size - df.loc[df.rep == n, 'lg'].iloc[-1] for n in df.rep]
        else:
            cutsdf = args[0]
        # ic(cutsdf)
        #iterate
        for n in df.rep:
            cutsdf[f"{n}"] = cutsdf['rest'] - df.loc[df.rep == n, 'lg'].iloc[-1]
        cutsdf = cutsdf.drop(columns=["rest"])
        ###REFORMAT ? NP.WHERE ?
        for n in cutsdf.index:
            for col in cutsdf.columns:
                if str(col) in n.split("_") : 
                    cutsdf.loc[n, col] = np.NaN
                if int(col) < int(n.split("_")[-1]):
                    cutsdf.loc[n, col] = np.NaN
        ####
        cutsdf[cutsdf < 0] = np.NaN
        cutsdf = cutsdf.dropna(how='all', axis = 0).dropna(how='all', axis = 1)
        return(cutsdf)

    def add_to_best_fits_recs(self, cutsdf, fit):
        fit = self.params["rules"][fit]
        for i in cutsdf.index:
            for n in cutsdf.columns:
                if cutsdf.loc[i, n] <fit:
                    self.temp[f"{i}_{n}"] = cutsdf.loc[i, n]
        # ic(self.temp)

    def add_depth_level(self, cutsdf, prod_df):
        d = {}
        cutsdf = pd.DataFrame(np.where(cutsdf < np.nanmin(prod_df.lg), np.NaN, cutsdf), index=cutsdf.index, columns=cutsdf.columns)

        # cutsdf = cutsdf.drop(columns=['rest'])
        cutsdf = cutsdf.dropna(how='all', axis = 0).dropna(how='all', axis = 1)
        for i in cutsdf.index:
            for n in cutsdf.columns:
                if cutsdf.loc[i, n] > 0:
                    d[f"{i}_{n}"] = cutsdf.loc[i, n]
        # ic(d)
        df = pd.DataFrame.from_dict(d, orient='index', columns=['rest'])
        return df if len(df)>0 else None
    
    def cut_best_fits(self, prod_df, fit):
        df = prod_df.copy()
        bfits = self.build_best_fits_df(df, fit)
        count = 1 if len(self.cuts) == 0 else self.cuts[-1].to_dict('records')[-1]['count'] +1
        while bfits is not None:
            self.make_best_cut(bfits, df, count)
            count +=1
            df = df.loc[~df.rep.isin(pd.concat(self.cuts).rep)]
            # ic(df)
            bfits = self.build_best_fits_df(df, fit)
        return df

    def make_best_cut(self, bfit, prod_df, count):
        best_cut = list(bfit.index)[0]
        df = prod_df.loc[prod_df.rep.isin([int(y) for y in list(best_cut.split("_")[1:])])].copy()
        df['count'] = count
        df['rest'] = self.params['load']['stock_lenght'] - df.lg.sum()
        df['origin'] = "new"
        df['excess'] = "waste"

        self.cuts.append(df)

    """STORAGE MANAGEMENT"""

    def choose_origin(self, prod_df):
        df = prod_df.copy()
        df['rest'] = self.params["load"]["stock_lenght"] - df.lg
        # ic(df)
        return df
    
    def cut_unsalvageable(self, prod_df):
        df = prod_df.copy()
        df = df.loc[df.rest<1000]
        df['origin'] = "new"
        df['excess'] = "waste"
        df['count'] = range(self.cuts[-1].to_dict('records')[-1]['count']+1, 
                            self.cuts[-1].to_dict('records')[-1]['count']+1+len(df))
        # ic(df)
        self.cuts.append(df)
        prod = prod_df.loc[~prod_df.rep.isin(self.cuts[-1].rep)]


        return prod

    def cut_stockables(self, prod_df):
        df = prod_df.copy()
        df['origin'] = "new"
        df['excess'] = "stock"
        df['count'] = range(self.cuts[-1].to_dict('records')[-1]['count']+1, 
                            self.cuts[-1].to_dict('records')[-1]['count']+1+len(df))
        # ic(df)
        self.cuts.append(df)
        prod = prod_df.loc[~prod_df.rep.isin(self.cuts[-1].rep)]
        return prod

    """STATS"""
    def get_prod_stats(self):
        df = pd.concat(self.cuts)
        print(df)
        linear_cut = np.sum(df.lg)/1000
        linear_waste = np.sum(df.loc[df.excess == "waste"].rest)/1000
        ic(linear_cut, linear_waste)
        part_waste = linear_waste/linear_cut*100
        print(f"WASTE PART : {round(part_waste,2)} %")

seq = Sequencer()

class Simulator:
    def __init__(self) -> None:
        pass

    def simulate_tube60_prod(self,nb_com):
        """Simulates a daily job

        Args:
            nb_com (int): how much orders

        Returns:
            Dataframe: prod df
        """        
        prod = [{
            "qtt" : random.choices(range(1,11), weights=[30,2,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.3])[0],
            "lg" : random.choice(range(1150,5700))
        } for n in range(nb_com)]
        df = pd.DataFrame.from_records(prod)
        df["lg"] = (5 * round(df['lg']/5))
        df["lg"] = df["lg"].astype(int)
        df["num_com"] = [f"1023-{n}" for n in list(random.sample(range(205,990), k = nb_com))]
        return df
    
sim = Simulator()
prod = sim.simulate_tube60_prod(40)
print("DAY ORDERS : ")
print(prod)
prod = seq.deploy_quantities(prod)
prod = seq.cut_best_fits(prod, fit='fit')
prod = seq.cut_best_fits(prod, fit="acceptable")
prod = seq.cut_best_fits(prod, fit="not_that_good")
prod = seq.choose_origin(prod)
prod = seq.cut_unsalvageable(prod)
prod = seq.cut_stockables(prod)
seq.get_prod_stats()

