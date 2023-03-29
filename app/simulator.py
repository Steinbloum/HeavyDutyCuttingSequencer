import pandas as pd
import numpy as np
import json
import random
from icecream import ic
from termcolor import cprint
from tabulate import tabulate


ic.configureOutput(includeContext=True)



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
            "lg" : random.choice(range(950,5700))
        } for n in range(nb_com)]
        df = pd.DataFrame.from_records(prod)
        df["lg"] = (5 * round(df['lg']/5))
        df["lg"] = df["lg"].astype(int)
        df["label"] = [f"1023-{n}" for n in list(random.sample(range(205,990), k = nb_com))]
        return df
    
    def simulate_output(self, cuts_df):
        for n in cuts_df['count'].unique():
            _df = cuts_df.loc[cuts_df['count'] == n].copy()
            if len(_df) < 2 :
                continue
            origin = _df.loc[_df.index[0], "origin"]
            origin = 'Neuf' if origin == "new" else f"Chute {origin}"
            excess = _df.loc[_df.index[-1], "excess"]
            excess = "Benne" if excess == "waste" else "Stock"

            cprint('NOUVELLE COUPE', "red", attrs=['bold', 'underline'])
            print("\n")
            cprint(f"Origine tube : ", "yellow", attrs=['bold', 'underline'])
            cprint(origin, attrs=['bold'])
            print("\n")
            cprint("Coupes :", attrs=['bold', 'underline'])
            print(tabulate(_df[["lg", "label"]], tablefmt="mixed_grid", showindex=False, headers=['Longueur', 'Numéro']))
            print("\n")
            cprint(f"Destination Chute :" ,"yellow", attrs=['bold', 'underline'])
            cprint(excess, attrs=['bold'])

            print("\n***\n")

    def compare_algos(self, seqs, days, nprod = 30):
        ls = []
        for n in range(days):
            prod = self.simulate_tube60_prod(nprod)
            for seq in seqs:
                seq.run_sequencer(prod)
                d = seq._get_stats()
                ic(seq.cuts, seq.storage.storage)
                d["day"] = n
                d["name"] = seq.name
                ls.append(d)
                ic(d)
                seq.prod = None
                seq.cuts = pd.DataFrame()
                seq.avlbl_cuts = None
        return pd.DataFrame.from_records(ls)
    
                


