from plot import plot_comp
import pandas as pd

x = pd.read_pickle('price-data-XBTUSD')



plot_comp(x,x)