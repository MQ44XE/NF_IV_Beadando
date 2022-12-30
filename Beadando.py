print("Szia Peti")
print("Szia Máté!")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import csv

# 5 multi asset:
    # equity = SPY
    # fixed income = AGG
    # commodity energy = USO
    # commodity raw materials = GLD
    # commodity food = DBA

df1 = pd.read_csv("SPY.csv", index_col=0)
df1.columns = [colname+"_SPY" for colname in df1.columns]
df2 = pd.read_csv("AGG.csv", index_col=0)
df2.columns = [colname+"_AGG" for colname in df2.columns]
df3 = pd.read_csv("USO.csv", index_col=0)
df3.columns = [colname+"_USO" for colname in df3.columns]
df4 = pd.read_csv("GLD.csv", index_col=0)
df4.columns = [colname+"_GLD" for colname in df4.columns]
df5 = pd.read_csv("DBA.csv", index_col=0)
df5.columns = [colname+"_DBA" for colname in df5.columns]
yield_curve = pd.read_csv("yield-curve.csv", index_col=0)

#adding returns to dataframe
df1["Daily_return_SPY"] = df1["Adj Close_SPY"] / df1["Adj Close_SPY"].shift(1) - 1
df2["Daily_return_AGG"] = df2["Adj Close_AGG"] / df2["Adj Close_AGG"].shift(1) - 1
df3["Daily_return_USO"] = df3["Adj Close_USO"] / df3["Adj Close_USO"].shift(1) - 1
df4["Daily_return_GLD"] = df4["Adj Close_GLD"] / df4["Adj Close_GLD"].shift(1) - 1
df5["Daily_return_DBA"] = df5["Adj Close_DBA"] / df5["Adj Close_DBA"].shift(1) - 1


#gathering returns into one dataframe
only_returns = pd.DataFrame()
only_returns["SPY"]=df1["Daily_return_SPY"]
only_returns["AGG"]=df2["Daily_return_AGG"]
only_returns["USO"]=df3["Daily_return_USO"]
only_returns["GLD"]=df4["Daily_return_GLD"]
only_returns["DBA"]=df5["Daily_return_DBA"]

#filtering nan values due to DBA
only_returns = only_returns.loc['2007-01-08':]

#mean and var-covar matrix
meanReturns = np.mean(only_returns, axis=0)
covReturns = np.cov(only_returns, rowvar=False)


# USD Govt yield curve 3 month
RiskFree = yield_curve["close"]
RiskFree = RiskFree.loc['2007-01-08':]
RiskFreeMeanDaily = np.mean(RiskFree)/252

# Sharpe mutató maximalizálás
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

# Csúszóablak függvény
def csuszo_ablak(only_returns, yield_curve, start_date, end_date):

    only_returns = only_returns.loc[start_date:end_date]
    yield_curve = yield_curve.loc[start_date:end_date]

    RiskFree = yield_curve["close"]
    RiskFreeMeanDaily = np.mean(RiskFree) / 252
    meanReturns = np.mean(only_returns, axis=0)
    covReturns = np.cov(only_returns, rowvar=False)

    def minimize_this_cs(weights, meanReturns, covReturns, RiskFreeMeanDaily):
        excess_return = np.matmul(np.array(meanReturns), weights.transpose()) - RiskFreeMeanDaily
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
    cons = ({'type': 'eq', 'fun': constraint_cs})
    for i in range(5):
        bounds.append((0, 1))

    optimization = optimize.minimize(minimize_this_cs, x0=x0, args=(meanReturns, covReturns, RiskFreeMeanDaily), method='SLSQP', bounds=bounds, constraints=cons, tol=10 ** -3)
    stand_dev = np.sqrt(np.matmul(np.matmul(optimization.x, covReturns), optimization.x.transpose()))
    port_return=np.matmul(np.array(meanReturns), optimization.x.transpose())
    return [optimization.fun*np.sqrt(252), optimization.x, stand_dev, port_return]

dates=only_returns.index
#print(dates[1258:])
print(csuszo_ablak(only_returns,yield_curve,'2007-01-08','2012-01-08'))

# 5 éves csúszóablak
sharpes=[]
w=[]
stand_dev=[]
port_return=[]

start_date_int = 1261
end_date_int = 1400

for i in range(start_date_int,end_date_int):
    [a,b,c,d]=csuszo_ablak(only_returns, yield_curve, dates[i-(5*252)], dates[i])
    sharpes.append(a)
    w.append(b)
    stand_dev.append(c)
    port_return.append(d)

# Sharpe csúszóablak plot

df_weights = pd.DataFrame(w)
df_weights.columns = ["SPY","AGG","USO","GLD","DBA"]
df_weights.index = dates[start_date_int:end_date_int]
df_weights.columns.names = ["ETFs"]
df_weights.plot()
plt.title("Weights of ETFs in Portfolio")
plt.xlabel("Dates")
plt.ylabel("Weights")
figure = plt.gcf()
figure.set_size_inches(10, 8)
#plt.savefig("weights.png", dpi=100)
plt.show()

df_std = pd.DataFrame(stand_dev)
df_std.index = dates[start_date_int:end_date_int]
df_std.columns = ["σ"]
df_std.plot()
plt.title("Daily Standard Deviation of the Portfolio")
plt.xlabel("Dates")
figure = plt.gcf()
figure.set_size_inches(10, 8)
#plt.savefig("Portfolio Daily Std.Dev.png", dpi=100)
plt.show()

df_ptfret = pd.DataFrame(port_return)
df_ptfret.index = dates[start_date_int:end_date_int]
df_ptfret.columns = ["r"]
df_ptfret.plot()
plt.title("Daily Returns of the Portfolio")
plt.xlabel("Dates")
figure = plt.gcf()
figure.set_size_inches(10, 8)
#plt.savefig("port_daily_return.png", dpi=100)
plt.show()

#####
#Maximum Drawdown
#####

#csúszó ablakos mdd optimalizáció
def csuszo_ablak_mdd(only_returns, start_date, end_date):
    only_returns = only_returns.loc[start_date:end_date]
    def minimize_this_mdd(weights, only_returns):
        portfolio_columnwise = only_returns * weights
        portfolio = pd.DataFrame()
        portfolio["Total"] = portfolio_columnwise.sum(axis=1)
        port_index = (1 + portfolio).cumprod()
        port_peaks = port_index.cummax()
        drawdown = (port_index - port_peaks) / port_peaks
        return -drawdown.min()
    def constraint_mdd(weights):
        sum = 0
        for weight in weights:
            sum = sum + weight
        return sum - 1

    bounds_mdd = []
    x0_mdd = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    cons_mdd = ({'type': 'eq', 'fun': constraint_mdd})
    for i in range(5):
        bounds_mdd.append((0, 1))

    optimization_mdd = optimize.minimize(minimize_this_mdd, x0=x0_mdd, args=(only_returns), method='SLSQP', bounds=bounds_mdd, constraints=cons_mdd, tol=10 ** -3)

    meanReturns = np.mean(only_returns, axis=0)
    covReturns = np.cov(only_returns, rowvar=False)
    stand_dev_mdd = np.sqrt(np.matmul(np.matmul(optimization_mdd.x, covReturns), optimization_mdd.x.transpose()))
    port_return_mdd = np.matmul(np.array(meanReturns), optimization_mdd.x.transpose())
    return [optimization_mdd.fun, optimization_mdd.x, stand_dev_mdd, port_return_mdd]

#egész időszakra vonatkozó mdd optimalizáció
[mdd_full_timeline, mdd_weights,c,d] = csuszo_ablak_mdd(only_returns,dates[1],dates[len(dates)-1])

#egész időszakra vonatkozó mdd optimalizáció PLOTOLÁS
mdd_port_columnwise = only_returns*mdd_weights
mdd_port=mdd_port_columnwise.sum(axis=1)
mdd_port_index=(1+mdd_port).cumprod()
mdd_port_peaks=mdd_port_index.cummax()
mdd_drawdown=(mdd_port_index-mdd_port_peaks)/mdd_port_peaks
mdd_port_index.plot()
plt.show()
mdd_drawdown.plot()
plt.show()

maximum_drawdown=[]
w_mdd=[]
stand_dev_mdd=[]
port_return_mdd=[]

for i in range(start_date_int,end_date_int):
    [a,b,c,d]=csuszo_ablak_mdd(only_returns, dates[i-(5*252)], dates[i])
    maximum_drawdown.append(a)
    w_mdd.append(b)
    stand_dev_mdd.append(c)
    port_return_mdd.append(d)

df_weights_mdd = pd.DataFrame(w_mdd)
df_weights_mdd.columns = ["SPY","AGG","USO","GLD","DBA"]
df_weights_mdd.index = dates[start_date_int:end_date_int]
df_weights_mdd.columns.names = ["ETFs"]
df_weights_mdd.plot()
plt.title("Weights of ETFs in Portfolio /w MDD")
plt.xlabel("Dates")
plt.ylabel("Weights")
figure = plt.gcf()
figure.set_size_inches(10, 8)
#plt.savefig("weights.png", dpi=100)
plt.show()

pass