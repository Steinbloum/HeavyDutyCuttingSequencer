from app.storage import Storage
import time
from termcolor import cprint
from icecream import ic
import pandas as pd
import numpy as np 
ic.configureOutput(includeContext=True)


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
def test_init_storage():
    s = Storage("tests/files/cfg.json")
    s.init_storage()
    sto = s.storage
    assert isinstance(sto, pd.DataFrame)
    assert list(sto.columns) == ["mini", "maxi", "qtt", "status", "name"]

@wrapper
def test_get_location_map():
    s = Storage("tests/files/cfg.json")
    s.init_storage()
    d = s._get_location_map_dict()
    assert isinstance(d, dict)
    df = pd.DataFrame({"a" : [1200, 3002, 2546, 8000]})
    ls  = [[k if x in v else False for x in df.a] for k, v in s.loc_mapping_dict['store'].items()]

    ic(ls)
    ls= [x for x in [y for y in ls if y]]
        
        # np.NaN for x in ls if any(x)]
    ic(ls)
    df['rt_location'] = [k if x in v else np.NaN for x in df.a for k, v in s.loc_mapping_dict['retrieve'].items()]
    assert df.st_location.tolist() == ["1m0", "3m0", "2m5"]
    assert df.rt_location.tolist() == ["1m5", "3m5", "3m0"]


