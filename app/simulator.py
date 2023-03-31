import pandas as pd
import numpy as np
import json
import random
from icecream import ic
from termcolor import cprint
from tabulate import tabulate
import plotly.graph_objects as go


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
        
        
        stats = []
        cuts = []
        storages = []
        params = [dict(seq.params, seq_name = seq.name) for seq in seqs]

        for n in range(days):
            prod = self.simulate_tube60_prod(nprod)
            for seq in seqs:
                ic(f"DAY {n}, SEQ {seq.name}")
                seq.run_sequencer(prod)
                stat = seq._get_stats()
                cut = seq.cuts.copy()
                storage = seq.storage.storage.copy()
                for _ in [stat, cut, storage]:
                    _['day'] = n
                    _['seq_name'] = seq.name
                
                stats.append(stat)
                cuts.append(cut)
                storages.append(storage)
                # ic(storage)
                seq.prod = None
                seq.cuts = pd.DataFrame()
                seq.avlbl_cuts = None
        return {"stats" : pd.DataFrame.from_records(stats),
                "cuts" : pd.concat(cuts),
                "storage" : pd.concat(storages), 
                "params" : params}
    
    def output_stats_histo(self, stats):

        ic(stats)
        ic(stats.groupby("seq_name").sum())
        # input()
        gr = stats.groupby("seq_name").sum()
        # ls = [stats.loc[stats.seq_name == x] for x in stats.seq_name.unique().tolist()]
        fig = go.Figure()
        fig.add_trace(go.Bar(name = "Orders",
                             y=gr.index,
                             x=gr.total_linear, 
                             opacity=0.5,
                             orientation='h'))
        fig.add_trace(go.Bar(name = "Stock Consumed",
                             y=gr.index,
                             x=gr.consumed_linear,
                             opacity=0.5,
                             orientation='h'))
        fig.add_trace(go.Bar(name = "Waste",
                             y=gr.index,
                             x=gr.total_waste + gr.total_linear,
                             opacity=0.5,
                             orientation='h',
                             marker_color = "black"))
        
        fig.update_layout(barmode = "overlay",
                          title_text="Material Consumption")
        fig.update_xaxes(title_text='Linear consumption (m)')
        fig.update_yaxes(title_text='Algo')
        fig.show()
                
    def output_activty_storage_lchart(self, cuts):
        ls = [cuts.loc[cuts.seq_name == x] for x in cuts.seq_name.unique().tolist()]

        fig = go.Figure()
        # ic(ls)
        for df in ls :
            color = f"rgb({','.join([str(x) for x in list(np.random.choice(range(100,200), size=3))])})"
            # ic(color)
            df = df.reset_index(drop=True)
            retrieved = df.dropna(subset = "origin"
                           ).loc[(~df.origin.isin(["new", "split"]))
                                 ].groupby("day").count().lg
            stored = df.dropna(subset = "excess"
                           ).loc[(~df.origin.isin(["waste"]))
                                 ].groupby("day").count().lg
            conc = pd.concat([retrieved, stored], axis=1).fillna(0)
            conc.columns = ["retrieved", "stored"]
            # ic(conc)
            conc = conc.sort_index()
            fig.add_trace(go.Scatter(name=df.seq_name.iloc[0],
                                x = conc.index,
                                y = conc.retrieved*-1,
                                line = dict(color = color)))
            fig.add_trace(go.Scatter(name=df.seq_name.iloc[0],
                                x = conc.index,
                                y = conc.stored,
                                line = dict(
                                    color = color
                                ),
                                showlegend=False))
        fig.update_layout(barmode='overlay',
                        title_text='Storage flow'
                        )
        fig.update_xaxes(title_text='Days')
        fig.update_yaxes(title_text='Flow')
        fig.show()
        # for storage in ls :
        #     for name in storage.name:
        #         fig.add_trace(go.Scatter(name = name,
        #                                 x = storage.day,
        #                                 y = storage.qtt))
        # fig.show()

    def add_price_and_time_info(self, cuts, params):
        df = cuts.copy()
        ic(df)
        ic(params)
        price = params['load']['linear_price']
        def get_value(lg):
            return lg*price/1000 if lg>0 else np.NaN
        
        def get_time(origin):
            cut_time = params['machine']['cutting_time']
            load_time = params['machine']['loading_time']
            if isinstance(origin, str):
                if origin == "split":
                    return cut_time
                else:
                    return load_time + cut_time
            else:
                return cut_time
        df['waste_value'] = [get_value(x) for x in df.rest]
        df['cut_time'] = [get_time(x) for x in df.origin]
        return df
    
    def output_general_stats(self, cuts, params):

        # ls = [cuts.loc[cuts.seq_name == x] for x in cuts.seq_name.unique().tolist()]
        recs = []                
        df = self.add_price_and_time_info(cuts, params)
        # ic(df)
        d = dict(time = f"{int(np.sum(df.cut_time)/3600)} hours",
            waste = f"{int(np.sum(df.loc[df.excess == 'waste', 'rest'])/1000)}m",
            ordered = f"{int(np.sum(df.lg)/1000)}m",
            consumed_units = f"{int(np.sum(df.origin.value_counts()['new']))} units",
            consumed_linear = f"{int(np.sum(df.origin.value_counts()['new'])*params['load']['stock_lenght']/1000)}m",
            waste_value = f"{int(np.sum(df.waste_value))} €",
            waste_ratio = f"{int(np.sum(df.loc[df.excess == 'waste', 'rest']))*100/int(np.sum(df.dropna(subset= 'label').lg))}",
            name = df.seq_name.iloc[0])
        # ic(d)
        recs.append(d)
        
        df = pd.DataFrame.from_records(recs)
        ic(df)
        df.index = df.seq_name
        ic(df)

