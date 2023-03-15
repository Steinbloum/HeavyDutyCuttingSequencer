# from app.seq2 import Hdcs
from app.seq2 import Hdcs
from app.sequencer import Simulator
from app.storage import Storage
import time
from termcolor import cprint
from icecream import ic
import pandas as pd
import numpy as np 
ic.configureOutput(includeContext=True)

sim = Simulator()
seq = Hdcs("tests/files/cfg.json")
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
def test_get_storage_info():
    seq.storage = Storage("tests/files/cfg.json")
    seq.storage.init_storage()
    prod = pd.read_csv("tests/files/prod_tst.csv", index_col=False)
    seq.prod = prod
    seq.init_cuts_df()
    df = seq.get_storage_info()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 14
    assert len(df.dropna()) == 6
    seq.storage.storage.loc[seq.storage.storage.name == "3m5", "status"] = "OPEN"
    
    df = seq.get_storage_info()
    assert "OPEN" in df.store_status.tolist()
    assert "OPEN" in df.rtv_status.tolist()

@printer
def test_add_depth():
    seq.storage = Storage("tests/files/cfg.json")
    seq.storage.init_storage()
    prod = pd.read_csv("tests/files/prod_tst.csv", index_col=False)
    seq.prod = prod
    seq.init_cuts_df()
    df = seq._add_depth()
    # ic(df)
    assert len(df) == 14
    assert len(df.columns) == 12

@printer
def test_make_combinations_list():
    seq.storage = Storage("tests/files/cfg.json")
    seq.storage.init_storage()
    prod = pd.read_csv("tests/files/prod_tst.csv", index_col=False)
    seq.prod = prod
    seq.init_cuts_df()
    df = seq._add_depth()
    df = seq._get_valid_combinations_dataframe(df)
    seq.avlbl_cuts = pd.concat([seq.avlbl_cuts, df])
    df = seq._add_depth()
    df = seq._get_valid_combinations_dataframe(df)
    seq.avlbl_cuts = pd.concat([seq.avlbl_cuts, df])
    df = seq._add_depth()
    df = seq._get_valid_combinations_dataframe(df)
    seq.avlbl_cuts = pd.concat([seq.avlbl_cuts, df])
    df = seq.avlbl_cuts
    assert len(df) == 106
    assert df.columns.tolist() == ["lg", "rest", "depth"]
    assert np.max(df.depth) == 4
    ls = [seq._get_int_list_from_reps(x) for x in df.index.tolist()]
    ls = ["_".join([str(x) for x in y]) for y in ls]
    assert len(list(dict.fromkeys(ls))) == len(ls)

@printer
def test_get_max_depth():
    seq.storage = Storage("tests/files/cfg.json")
    seq.storage.init_storage()
    seq.prod = pd.read_csv("tests/files/prod_tst.csv", index_col=False)
    # seq._deploy_quantities()
    # ic(len(seq.prod))
    seq.init_cuts_df()
    df = seq.get_max_depth_combinations()
    assert len(df) == 106
    assert df.columns.tolist() == ['lg', 'rest', 'depth', 'store_loc', 'store_status', 'rtv_loc', 'rtv_status']
    assert np.max(df.depth) == 4
    ls = [seq._get_int_list_from_reps(x) for x in df.index.tolist()]
    ls = ["_".join([str(x) for x in y]) for y in ls]
    assert len(list(dict.fromkeys(ls))) == len(ls)

@printer
def test_make_cut():

    seq.storage = Storage("tests/files/cfg.json")
    seq.storage.init_storage()
    seq.prod = pd.read_csv("tests/files/prod_tst.csv", index_col=False)
    # ic(len(seq.prod))
    seq.init_cuts_df()
    df = seq.get_max_depth_combinations()
    # ic(df)
    seq.make_cut("_0_1_2_3")
    #test with storage

    for i in range(4):
        assert i not in seq.prod.index
        assert i in seq.cuts.rep

@printer
def test_cut_forced_waste():
    seq = Hdcs("tests/files/cfg.json")
    seq.storage = Storage("tests/files/cfg.json")
    seq.storage.init_storage()
    seq.prod = pd.read_csv("tests/files/prod_tst.csv", index_col=False)
    seq.init_cuts_df()
    seq.get_max_depth_combinations()
    seq.cut_forced_waste()
    reps = [12,13]
    assert seq.cuts.rep.tolist() == reps
    for x in seq.prod.rep:
        assert x not in reps
    
@printer
def test_cut_critical_retrieve():
    seq = Hdcs("tests/files/cfg.json")
    seq.storage = Storage("tests/files/cfg.json")
    seq.storage.init_storage()
    seq.storage.storage.loc[seq.storage.storage.name.isin(["3m5", "2m5"]), "qtt"] = 20
    seq.storage._update_status()
    ic(seq.storage.storage)
    # seq.prod = pd.read_csv("tests/files/prod_tst.csv", index_col=False)
    prod = sim.simulate_tube60_prod(35)
    seq.init_prod(prod)
    seq.init_cuts_df()
    seq.get_max_depth_combinations()
    seq.cut_forced_waste()
    seq._retrieve_critical()
    seq._store_critical()
    seq._cut_good_fits()
    seq._cut_good_fits(500)
    ic(seq.cuts)
    ic(seq.prod)
    ic(seq.storage.storage)





    # seq.cut_critical_retrieve()
