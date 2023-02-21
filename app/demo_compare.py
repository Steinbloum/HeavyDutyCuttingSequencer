from sequencer import Sequencer, Simulator
import pandas as pd
from icecream import ic
import numpy as np
from tabulate import tabulate
from termcolor import cprint

seq = Sequencer()
sim = Simulator()
orig = sim.simulate_tube60_prod(30)
ls = []
print("DAY ORDERS : ")
prod = orig.copy()
print(tabulate(prod, headers=prod.columns, tablefmt="rst"))
input('PRESS ENTER TO RUN ALGO')
def make_sequence(prod, _mode, seq):
    if mode == "max_concat" : 
        prio = ['cut_count', 'rest']
        ascendi = [False, True]
    elif mode == 'longest' : 
        prio = ["max_lg", "rest"]
        ascendi = [False, True]
    elif mode == "simple_best":
        prio = 'rest'
        ascendi = True
    prod = seq.deploy_quantities(prod)
    prod = seq.cut_best_fits(prod, fit='fit', priority=prio, ascend=ascendi, show=False)
    prod = seq.cut_best_fits(prod, fit="acceptable", priority=prio, ascend=ascendi, show=False)
    prod = seq.cut_best_fits(prod, fit="not_that_good", priority=prio, ascend=ascendi, show=False)
    prod = seq.choose_origin(prod)
    prod = seq.cut_unsalvageable(prod)
    prod = seq.cut_stockables(prod)
    res = seq.get_prod_stats(show = False)
    return dict(res, mode = _mode)
def convert_time(s):
    return f"{s/60}m"

mode = 'max_concat'

d = make_sequence(prod , mode, seq)
d["stock_lg"] = seq.params['load']['stock_lenght']
df = pd.DataFrame(d, index = [0])
ic(df)
ls.append(df)


seq2 = Sequencer()
seq2.params['load']['stock_lenght'] = 8000
seq2.params['machine']['loading_time'] = 150
cprint('Changed stock lenght to 8m and machine loading time to 150sec', attrs=["bold"])
input()

prod = orig.copy()
d = make_sequence(prod , mode, seq2)
d["stock_lg"] = seq2.params['load']['stock_lenght']
# ic(d)
df = pd.DataFrame(d, index=[0])
ls.append(df)
# ic(ls)


# df = pd.concat([pd.DataFrame.from_dict(x, orient = "columns", index=[0]) for x in ls])
df = pd.concat(ls, ignore_index=True)
df['total_time'] = [convert_time(x) for x in df.total_time]

print(tabulate(df, headers=df.columns, tablefmt="rst"))



