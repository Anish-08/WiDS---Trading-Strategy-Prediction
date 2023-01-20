import pandas as pd
from sklearn import preprocessing

class main:
    '''This is the class for storing and processing all the data 
    
    :param data: This is the main trading data 
    :type data: pandas.core.frame.DataFrame
    '''
    def __init__(self,filename):
        self.data: pd.DataFrame = pd.read_pickle(filename)
        self.add_fields()
        self.drop_fields()
        #self.rescale()

    def drop_fields(self):
        
        self.data = self.data.drop(['symbol','timestamp','vwap','lastSize','turnover','homeNotional','foreignNotional'], axis =1)

        self.data = self.data.dropna(axis=1)
        
        self.data = self.data.astype(float)

    def rescale(self):
        '''Rescale the data '''
        

        names = self.data.columns
        scaler = preprocessing.StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        self.data = pd.DataFrame(scaled_data, columns=names)

        arr = preprocessing.scale(self.data)
        self.data = pd.DataFrame(arr, columns=self.data.columns )

    def add_fields(self):
        '''Adding all the extra fields to help with prediction '''
        for n in [10,50,100,200]:
            self.SMA(n)
            self.EWMA(n)

        self.bbands(50)
        self.Force_index()
        self.CCI(20)
        self.EVM(14)
        self.ROC(5)

        


    def SMA(self, n, price_type='close'):
        '''Finding Moving average over n days and appending this result to our data
        
        :param n: Number of days over which moving average needs to be taken
        :type n: int
        :param price_type: The type of price which we want to work with, close, open, high or low
        :type price_type: str
        '''

        sma  =  pd.Series(self.data[price_type].rolling(window=n,min_periods=0).mean(),name=str(n)+"-rolling mean")
        self.data = self.data.join(sma)

    def EWMA(self, n, price_type='close'):
        '''Finding Exponential Moving average over n days and appending this result to our data
        
        :param n: The smoothing factor will be 2/(n+1)
        :type n: int
        :param price_type: The type of price which we want to work with, close, open, high or low
        :type price_type: str
        '''

        emwa = pd.Series(self.data[price_type].ewm(span=n).mean(), name=str(n)+'-rolling expo mean')
        self.data = self.data.join(emwa)

    def bbands(self, n, price_type='close'):
        '''Finding upper and lower Boillinger bands for our data which is rolling mean + 2* standard deviation and rollling mean - 2* standard deviation
        
        :param n: Number of days over which the band is to be calculated
        :type n: int
        :param price_type: The type of price which we want to work with, close, open, high or low
        :type price_type: str
        '''

        b_mean = self.data[price_type].rolling(window=n,min_periods=0).mean()
        b_std = self.data[price_type].rolling(window=n,min_periods=0).std()
        b_std[0]=0
        self.data['UpperB'] = b_mean+2*b_std
        self.data['LowerB'] = b_mean-2*b_std

    def Force_index(self):
        '''Finding force index which is difference in closing price times total volume traded 
        '''
        diff = self.data['close'].diff(1)
        diff[0] = 0
        FI = pd.Series(diff*self.data['volume'], name='Force Index')
        self.data = self.data.join(FI)

    def CCI(self,n):
        '''Finding the Commodity Channel Index which is (typical price - moving average)/Mead deviation*0.015
        :param n: Number of days over which the moving average is to be taken for CCI calculation
        :type n: int
        '''

        TP = (self.data['high']+self.data['low']+self.data['close'])/3
        CCI = pd.Series((TP-TP.rolling(window=n,min_periods=0).mean())/(0.015 * TP.rolling(window=n,min_periods=0).std()),name='CCI').fillna(0)
        #The fillna(0) is required to avoid NaN values
        self.data = self.data.join(CCI)

    def EVM(self, n, C=100000000):
        '''To calculate the Ease of Movement Indicator
        :param n: Number of days over which the moving average is to be taken for CCI calculation
        :type n: int
        :param C: The constant involved in calculation of EVM , usually 10^6
        '''
        diff = (self.data['high']+self.data['low']-self.data['high'].shift(1)- self.data['low'].shift(1))/2
        denom = (self.data['volume']/C)/(self.data['high']-self.data['low'])
        EVM = diff/denom
        EVM_MA = pd.Series(EVM.rolling(window=n, min_periods=0).mean(), name='EVM')
        self.data = self.data.join(EVM_MA)

    def ROC(self, n, price_type='close'):
        '''This calculates the percent rate of change over n days

        :param n: Number of days over which the band is to be calculated
        :type n: int
        :param price_type: The type of price which we want to work with, close, open, high or low
        :type price_type: str
        '''
        N = self.data[price_type].diff(n)
        D = self.data[price_type].shift(n)
        ROC = pd.Series(N/D , name='ROC')
        ROC = ROC.fillna(0)
        self.data = self.data.join(ROC)

    def print_data(self):
        self.data.to_csv('test.csv')
        