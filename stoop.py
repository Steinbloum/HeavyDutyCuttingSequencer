import pandas as pd

ls = ["10", "111", "89"]
ls2 = ["100", "111", "898"]
if any([x==y for x in ls for y in ls2]):
    print("yes")
print("10" in ls)

