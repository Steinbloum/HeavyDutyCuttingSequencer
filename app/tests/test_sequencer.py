# from app.seq2 import Hdcs
from app.seq2 import Hdcs
from app.storage import Storage
import time
from termcolor import cprint
from icecream import ic
import pandas as pd
import numpy as np 
ic.configureOutput(includeContext=True)

seq = Hdcs("tests/files/cfg.json")
def wrapper(func):
    """Prints the name of the tested function, 
        Prints the exec time of the test"""
    def wrap(*args, **kwargs):
        st = time.time()
        print("\n")
        lsn = func.__name__.split('_')
        # print(lsn)
        nm = ' '.join(lsn[1:]) if lsn[0] == "test" else ' '.join(lsn)
        cprint(f"Testing {nm}", "green", attrs=['underline'])
        func(*args, **kwargs)
        nd = round(time.time()-st,4)
        cprint(f"\nTest executed in {nd} seconds","green",  attrs=['underline'])
    return wrap

@wrapper
def test_get_storage_info():
    seq.storage = Storage("tests/files/cfg.json")
    seq.storage.init_storage()
    prod = pd.read_csv("tests/files/prod_tst.csv", index_col=False)
    seq.prod = prod
    seq.init_cuts_df()
    df = seq.get_storage_info()
    assert isinstance(df, pd.DataFrame)
    assert len(df == 18)
    assert len(df.dropna() == 13)
    seq.storage.storage.loc[seq.storage.storage.name == "2m0", "status"] = "OPEN"
    df = seq.get_storage_info()
    assert "OPEN" in df.store_status.tolist()
    assert "OPEN" in df.rtv_status.tolist()

@wrapper
def test_add_depth():
    seq.storage = Storage("tests/files/cfg.json")
    seq.storage.init_storage()
    prod = pd.read_csv("tests/files/prod_tst.csv", index_col=False)
    seq.prod = prod
    seq.init_cuts_df()
    seq._add_depth()


