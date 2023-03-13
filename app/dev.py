import pandas as pd
from sequencer import Simulator
from seq2 import Hdcs

s = Simulator()
seq = Hdcs()

# prod = s.simulate_tube60_prod(10)
# seq.init_prod(prod)
# print(seq.prod)
# seq.prod.to_csv("prod_tst.csv", index = False)
print(seq.storage.storage)
prod = s.simulate_tube60_prod(50)

seq.prod = prod
seq.init_cuts_df()
avlbl = seq.avlbl_cuts
avlbl['store_loc'] = [seq.storage.get_unique_location(x) for x in avlbl.lg]
avlbl["store_status"] = [seq.storage.get_status(x) for x in avlbl.store_loc]
avlbl['rtv_loc'] = [seq.storage.get_unique_location(x, "retrieve") for x in avlbl.lg]
avlbl["rtv_status"] = [seq.storage.get_status(x) for x in avlbl.rtv_loc]


