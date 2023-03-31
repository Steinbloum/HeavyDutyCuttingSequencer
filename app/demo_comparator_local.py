import pandas as pd
from simulator import Simulator
from seq2 import Hdcs
from icecream import ic
import itertools
import numpy as np

s = Simulator()
seq8 = Hdcs("app/tests/files/cfg_8m.json")
seq6 = Hdcs("app/tests/files/cfg.json")
seq7 = Hdcs("app/tests/files/cfg_7m.json")
seqsm = Hdcs("app/tests/files/cfg_6m_small_step.json")
seq8.name = "8m"
seq7.name = "7m"
seq6.name = "6m"
seqsm.name = "SS"


prod = s.simulate_tube60_prod(20)
ic(prod)
input()


# d = s.compare_algos([seq8,seq7, seq6, seqsm], 20, 15)
s.output_stats_histo(pd.read_csv('comp_stats.csv', index_col=False))
s.output_activty_storage_lchart(pd.read_csv('comp_cuts.csv', index_col=False))

