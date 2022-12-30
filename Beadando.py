print("Szia Peti")
print("Szia Máté!")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

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
yield_curve = pd.read_csv("yield-curve.csv", index_col=0)

#adding returns to dataframe
df1["Daily_return_SPY"] = df1["Adj Close_SPY"] / df1["Adj Close_SPY"].shift(1) - 1
df2["Daily_return_AGG"] = df2["Adj Close_AGG"] / df2["Adj Close_AGG"].shift(1) - 1
df3["Daily_return_USO"] = df3["Adj Close_USO"] / df3["Adj Close_USO"].shift(1) - 1
df4["Daily_return_GLD"] = df4["Adj Close_GLD"] / df4["Adj Close_GLD"].shift(1) - 1
df5["Daily_return_CORN"] = df5["Adj Close_CORN"] / df5["Adj Close_CORN"].shift(1) - 1


#gathering returns into one dataframe
only_returns = pd.DataFrame()
only_returns["SPY"]=df1["Daily_return_SPY"]
only_returns["AGG"]=df2["Daily_return_AGG"]
only_returns["USO"]=df3["Daily_return_USO"]
only_returns["GLD"]=df4["Daily_return_GLD"]
only_returns["CORN"]=df5["Daily_return_CORN"]

#filtering nan values due to CORN
only_returns = only_returns.loc['2010-06-10':]

#mean and var-covar matrix
meanReturns = np.mean(only_returns, axis=0)
covReturns = np.cov(only_returns, rowvar=False)


# USD Govt yield curve 3 month
RiskFree = yield_curve["close"]
RiskFree = RiskFree.loc['2010-06-10':]
RiskFreeMeanDaily = np.mean(RiskFree)/252

def minimize_this(weights, meanReturns, covReturns, RiskFreeMeanDaily):
    excess_return = np.matmul(np.array(meanReturns), weights.transpose()) - RiskFreeMeanDaily
    stand_dev = np.sqrt(np.matmul(np.matmul(weights, covReturns), weights.transpose()))
    negative_sharpe = -(excess_return / stand_dev)
    return negative_sharpe

def constraint(weights):
    sum = 0
    for weight in weights:
        sum = sum + weight
    return sum - 1

bounds=[]
x0 = np.array([0.2,0.2,0.2,0.2,0.2])
cons = ({'type': 'eq', 'fun': constraint})
for i in range(5):
    bounds.append((0,1))

optimization = optimize.minimize(minimize_this, x0=x0, args=(meanReturns, covReturns, RiskFreeMeanDaily), method='SLSQP', bounds=bounds, constraints=cons, tol=10 ** -3)
print(optimization.fun*np.sqrt(252))
print(optimization.x)

def csuszo_ablak(only_returns, yield_curve, start_date, end_date):

    only_returns = only_returns.loc[start_date:end_date]
    yield_curve = yield_curve.loc[start_date:end_date]

    meanReturns = np.mean(only_returns, axis=0)
    meanYC = np.mean(yield_curve, axis=0)
    covReturns = np.cov(only_returns, rowvar=False)

    RF_daily = meanYC / 252

    def minimize_this_cs(weights, meanReturns, covReturns, RF_daily):
        excess_return = np.matmul(np.array(meanReturns), weights.transpose()) - RF_daily
        stand_dev = np.sqrt(np.matmul(np.matmul(weights, covReturns), weights.transpose()))
        negative_sharpe = -(excess_return / stand_dev)
        return negative_sharpe

    def constraint_cs(weights):
        sum = 0
        for weight in weights:
            sum = sum + weight
        return sum - 1

    bounds = []
    x0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    cons = ({'type': 'eq', 'fun': constraint})
    for i in range(5):
        bounds.append((0, 1))

    optimization = optimize.minimize(minimize_this, x0=x0, args=(meanReturns, covReturns, RF_daily), method='SLSQP',
                                     bounds=bounds, constraints=cons, tol=10 ** -3)
    return [optimization.fun, optimization.x]

dates=only_returns.index
#print(dates[1258:])


pass
