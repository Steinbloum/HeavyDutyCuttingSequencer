import pandas as pd
from sequencer import Simulator
from seq2 import Hdcs
from icecream import ic
import itertools

s = Simulator()
seq = Hdcs()


# prod = s.simulate_tube60_prod(60)
# # ic(prod)
# seq.init_prod(prod)
# seq.init_cuts_df()
# seq.get_max_depth_combinations()
# seq.avlbl_cuts.to_csv("large_prod.csv", index=)
seq.prod = pd.read_csv("large_prod.csv", index_col="Unnamed: 0")
seq.avlbl_cuts = pd.read_csv("large_available.csv", index_col="Unnamed: 0")
# ic(len(seq.avlbl_cuts))
seq.cut_forced_waste()
# seq.update_avlbl_cuts()
# ic(seq._get_unstorable_list())
df = seq._get_priority_cuts()
ic(df, seq.cuts, seq.prod)
while len(df)>0:
    df = seq.make_cut(df.index[0], update=df)
ic(df, seq.cuts, seq.prod, len(seq.avlbl_cuts))
seq.update_avlbl_cuts()
ic(len(seq.avlbl_cuts), seq.avlbl_cuts.sort_values(by=["rest", "depth"], ascending=[True, False]), seq.prod)
seq._cut_good_fits()
ic(len(seq.avlbl_cuts), seq.avlbl_cuts.sort_values(by=["rest", "depth"], ascending=[True, False]), seq.prod)
seq._cut_good_fits(seq.params['storage']['step'])
ic(len(seq.avlbl_cuts), seq.avlbl_cuts.sort_values(by=["rest", "depth"], ascending=[True, False]), seq.prod)
seq._retrieve_max()
ic(len(seq.avlbl_cuts), seq.avlbl_cuts.sort_values(by=["rest", "depth"], ascending=[True, False]), seq.prod)
ic(seq.cuts)
seq._store_max()
ic(len(seq.avlbl_cuts), seq.avlbl_cuts.sort_values(by=["rest", "depth"], ascending=[True, False]), seq.prod)
ic(seq.cuts)
ic(seq.storage.storage)
ic(seq.avlbl_cuts)
for n in seq.avlbl_cuts.index:
    seq.make_cut(n)
ic(len(seq.avlbl_cuts), seq.avlbl_cuts.sort_values(by=["rest", "depth"], ascending=[True, False]), seq.prod)
ic(seq.cuts)
ic(seq.storage.storage)





