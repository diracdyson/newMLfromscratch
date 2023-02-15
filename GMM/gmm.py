import numpy as np
import pandas as pd

class gmm():
    def __init__(self ,x: pd.DataFrame(),y: np.array(dtype=np.float64):
        y=None
        self.x= x 
        self.mu=mu
        self.sigma= sigma


    def gauss(self):
        k = len(self.mu)
        sigma= np.diag(self.sigma )
        X =self.X - self.mu.T
        # mean deviated data 
       self.probs = (1/2*np.pi)**(k/2) *(np.linalg.det(sigma)**(0.5))*np.exp(-0.5*np.sum(X @ np.linalg.pinv(sigma)*X,axis=1))


        return self
    
    def graph(self,col='Hedge Fund',epsilon = 0.0001):
        fig,ax= plt.subplots()
        ax.scatter(self.x.index, self.x[col].values.reshape(-1,1),c =self.probs,label="GMM anomaly probs")
        ds = np.where(self.probs < epsilon)
        # mark outliers with threshold of epsilon
        ax.scatter(x.index[ds],self.x.iloc[ids,0],marker="O")

        ax.legend()
        plt.show()




