print("Szia Peti")
print("Szia Máté!")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 5 multi asset:
    # equity = SPY
    # fixed income = AGG
    # commodity energy = USO
    # commodity raw materials = GLD
    # commodity food = CORN

df1 = pd.read_csv("SPY.csv", index_col=0)
df1.columns = [colname+"_SPY" for colname in df1.columns]
df2 = pd.read_csv("AGG.csv", index_col=0)
df2.columns = [colname+"_AGG" for colname in df2.columns]
df3 = pd.read_csv("USO.csv", index_col=0)
df3.columns = [colname+"_USO" for colname in df3.columns]
df4 = pd.read_csv("GLD.csv", index_col=0)
df4.columns = [colname+"_GLD" for colname in df4.columns]
df5 = pd.read_csv("CORN.csv", index_col=0)
df5.columns = [colname+"_CORN" for colname in df5.columns]


pass
