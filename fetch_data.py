import bitmex
import time
import json
import requests
import pandas as pd
x =1 

bitmex_api_key_test = "" 
bitmex_api_secret_test = ""

client = bitmex.bitmex(api_key=bitmex_api_key_test, api_secret=bitmex_api_secret_test)

binSize='1d'
u  = client.Trade.Trade_getBucketed(binSize=binSize, count=1000, symbol='XBTUSD', reverse=False).result()[0]
    

data = pd.DataFrame(u)
print(data)
data.to_pickle('data-XBTUSD')