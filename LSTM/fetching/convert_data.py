import pandas as pd 

def get_mid(x):


    x = x.drop(['symbol','vwap','lastSize','turnover','homeNotional','foreignNotional'],axis=1)

    first = x['timestamp']

    x = x.drop(['timestamp'],axis = 1)

    first = pd.Series([u.date() for u in first], name='date')

    x = x.join(first)

    x = x[['date','open','high','low','close','trades','volume']]

    high_prices = x['high']
    low_prices = x['low']


    mid = (high_prices + low_prices)/2
    for x in range(len(mid)):
        if x!=0 and x!= len(mid)-1:
            if (mid[x]-mid[x+1])>1000:
                mid[x] = (mid[x-1]+mid[x+1])/2

    return list(mid)

u = pd.read_pickle("new-data-1")
u = get_mid(u)
v = pd.read_pickle("new-data-2")
v = get_mid(v)
w = pd.read_pickle("new-data-3")
w = get_mid(w)

u.extend(v)

u.extend(w)

print(u)

mid = pd.Series(u)





pd.to_pickle(mid,'price-data-XBTUSD')

