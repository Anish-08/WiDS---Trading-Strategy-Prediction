from trading_data_class import main
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import xgboost as xgb

class Model:
    def __init__(self):
        data = main('data-XBTUSD')
        self.X: pd.DataFrame = data.data
        self.train_size = 0
        self.Y: pd.DataFrame = None
        self.getY()
        
        self.X_train, self.Y_train, self.X_test, self.Y_test = self.get_split()

        self.model = None


    def getY(self, price_type='open'):
        Y = []
        prices = list(self.X[price_type])
        for i in range(len(prices)-1):
            if prices[i+1]>prices[i]:
                Y.append(1)
            else:
                Y.append(0)

        self.X = self.X[:-1]
        self.Y = np.array(Y)

    def get_split(self, ratio=0.75):
        self.train_size = int(ratio*self.X.shape[0])
        return self.X[:self.train_size],self.Y[:self.train_size],self.X[self.train_size:],self.Y[self.train_size:]

    def train(self):
        self.model =  xgb.XGBClassifier()
        self.model.fit(self.X_train,self.Y_train)

    def predict(self):
        size = self.Y_test.shape[0]
        acc = 0
        tot = 0
        for i in range(size):
            y_pred = self.model.predict(self.X[:self.train_size+i])
            
            if round(y_pred[self.train_size+i-1])==self.Y[self.train_size+i-1]:
                acc+=1
            tot+=1
        return acc/tot
        

if __name__=='__main__':   
    M = Model()
    print(f'Starting training')
    M.train()
    print(f'Training ended')
    u = M.predict()
    print(f'Accuracy is {100*u}')


