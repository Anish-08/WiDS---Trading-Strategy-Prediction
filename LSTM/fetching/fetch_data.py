import bitmex
import time
import json
import requests
import pandas as pd
import datetime

x = datetime.datetime(2016, 1, 1)
y = datetime.datetime(2019, 1, 31)
z = datetime.datetime(2021, 10, 27)
#print(x)

bitmex_api_key_test = ""
bitmex_api_secret_test = ""

client = bitmex.bitmex(api_key=bitmex_api_key_test, api_secret=bitmex_api_secret_test)

binSize='1d'
u  = client.Trade.Trade_getBucketed( binSize=binSize, startTime = z, count=1000, symbol='XBTUSD', reverse=False).result()[0]

u = pd.DataFrame(u)

print(u)
u.to_pickle('new-data-3')