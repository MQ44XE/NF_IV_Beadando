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

#adding returns to dataframe
df1["Daily_return_SPY"] = df1["Close_SPY"] / df1["Close_SPY"].shift(1) - 1
df2["Daily_return_AGG"] = df2["Close_AGG"] / df2["Close_AGG"].shift(1) - 1
df3["Daily_return_USO"] = df3["Close_USO"] / df3["Close_USO"].shift(1) - 1
df4["Daily_return_GLD"] = df4["Close_GLD"] / df4["Close_GLD"].shift(1) - 1
df5["Daily_return_CORN"] = df5["Close_CORN"] / df5["Close_CORN"].shift(1) - 1


pass
