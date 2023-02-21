from sequencer import Sequencer, Simulator
import pandas as pd
from icecream import ic
import numpy as np
from tabulate import tabulate

seq = Sequencer()
sim = Simulator()
orig = sim.simulate_tube60_prod(40)
ls = []
print("DAY ORDERS : ")
prod = orig.copy()
print(tabulate(prod, headers=prod.columns, tablefmt="rst"))
input('PRESS ENTER TO RUN ALGO')
def make_sequence(prod, _mode):
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
    prod = seq.cut_best_fits(prod, fit='fit', priority=prio, ascend=ascendi, show=True)
    prod = seq.cut_best_fits(prod, fit="acceptable", priority=prio, ascend=ascendi, show=True)
    prod = seq.cut_best_fits(prod, fit="not_that_good", priority=prio, ascend=ascendi, show=True)
    prod = seq.choose_origin(prod)
    prod = seq.cut_unsalvageable(prod)
    prod = seq.cut_stockables(prod)
    res = seq.get_prod_stats(show = True)
    return dict(res, mode = _mode)

mode = 'max_concat'

d = make_sequence(prod , mode)
df = pd.DataFrame.from_dict(d, orient = 'index')
df[0] = [round(x, 2) if isinstance(x, float) else x for x in df[0]]
# print(df)
print(tabulate(df))
input('PRESS ENTER TO SEE OUTPUT')
print("\n\n")
sim.simulate_output(pd.concat(seq.cuts))