import pandas as pd
from simulator import Simulator
from seq2 import Hdcs
from icecream import ic
import itertools
import numpy as np

s = Simulator()
seq8 = Hdcs("app/tests/files/cfg_8m.json")
seq6 = Hdcs("app/tests/files/cfg.json")
seq8.name = "8m"
seq6.name = "6m"





# seq.prod = pd.read_csv("prod_tst splw.csv", index_col=False)
# seq.init_cuts_df()
df = s.compare_algos([seq8, seq6], 20, 20)
ic(df)
df.to_csv("compare.csv", index=False)

# def split_waste(cut, storage):
#     rest = cut.rest.iloc[-1]
#     st = storage.loc[storage.status != "FULL"].sort_values(by="qtt")
#     if len(st) == 0:
#         return cut
#     ic(st)
#     while rest > np.min(st.mini):
#         minis = st.mini.to_list()
#         for n in minis:
#             if rest >= n:
#                 split = {
#                     "lg" : n,
#                     "label" : np.NaN,
#                     "rep" : np.NaN,
#                     "depth" : cut.depth.iloc[-1],
#                     "origin" : "split",
#                     "excess" : st.loc[st.mini == n, "name"].iloc[0],
#                     "rest" : rest - n,
#                     "cnt" : cut.cnt.iloc[-1]
#                 }

#                 cut.loc[cut.index[-1], "rest"] = np.NaN
#                 cut = pd.concat([cut, pd.DataFrame(split, index=[0])], ignore_index=True)
#                 rest = cut.rest.iloc[-1]
#     cut.loc[cut.index[-1], "excess"] = st.loc[st.mini == cut.lg.iloc[-1], "name"].iloc[0]
#     return cut
    


# ic(split_waste(seq.cuts, seq.storage.storage))

