from app.storage import Storage
import time
from termcolor import cprint
from icecream import ic
import pandas as pd
import numpy as np 
ic.configureOutput(includeContext=True)


def printer(func):
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



@printer
def test_init_storage():
    s = Storage("tests/files/cfg.json")
    s.init_storage()
    sto = s.storage
    assert isinstance(sto, pd.DataFrame)
    assert list(sto.columns) == ["mini", "maxi", "qtt", "status", "name"]

@printer
def test_get_location_map():
    s = Storage("tests/files/cfg.json")
    s.init_storage()
    d = s._get_location_map_dict()
    assert isinstance(d, dict)
    df = pd.DataFrame({"a" : [1200, 3002, 2546, 8000]})
    df["store_loc"] = [s.get_unique_location(x) for x in df.a]
    df["rtv_loc"] = [s.get_unique_location(x, "retrieve") for x in df.a]
    assert df.store_loc.tolist() == ["1m0", "3m0", "2m5", np.NaN]
    assert df.rtv_loc.tolist() == ["1m5", "3m5", "3m0", np.NaN]

@printer
def test_get_status_map():
    
    s = Storage("tests/files/cfg.json")
    s.init_storage()
    df = pd.DataFrame({"a" : [1200, 3002, 2546, 8000]})
    df["store_loc"] = [s.get_unique_location(x) for x in df.a]
    df['status'] = [s.get_status(n) for n in df.store_loc]
    s.storage.loc[s.storage.name == "1m0", "status"] = "OPEN"
    df['status'] = [s.get_status(n) for n in df.store_loc]
    assert df.loc[df.store_loc == "1m0", "status"].iloc[0] == "OPEN"

def test_increment_decrement_storage():
    s = Storage("tests/files/cfg.json")
    s.init_storage()
    # ic(s.storage)
    for n in range(9):
        s.store('1m5')
    # ic(s.storage)
    for n in range(11):
        s.store("1m5")
    # ic(s.storage)
    for n in range(12):
        s.retrieve('1m5')




# class Test