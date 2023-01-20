from matplotlib import pyplot as plt

def plot_comp(data):
    data = list(data)
    #pred  = list(pred)
    #size = max(len(data),len(pred))
    plt.figure(figsize = (18,9))
    plt.plot(range(len(data)),data,color='b')
    #plt.plot(range(size),pred,color='orange',label='Prediction')
    plt.xlabel('Date')
    plt.ylabel('Mid Price')
    plt.savefig('Smoothened-Normalized-Training-Data')