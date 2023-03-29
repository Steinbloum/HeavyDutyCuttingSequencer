import pandas as pd
import random

minis = [1000,1500,2000,2500,3000,3500]
def split_waste(lg):
    if lg < minis[1]:
        return False
    else : 
        for m in minis:
            if lg-m > 0:
                return m, lg-m


ls = [random.randint(0,4000) for n in range(100)]
for lg in ls:
    print(lg)
    print(split_waste(lg))
    print("*")