import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LogisticRegression():
    def __init__(self, x, y, alpha=0.1, max_iter=100, w0=None, verbos=False) -> None:
        if w0:
            self.w0 = w0
        else:
            self.w0 = np.zeros([x.shape[0], 1], dtype='float')
        self.x = x
        self.y = y
        self.alpha = alpha
        self.max_iter = max_iter
        self.wt = self.w0
        self.N = x.shape[0]
        self.verbos = verbos
    
    def __logistic(self, x, w):
        y_hat = 1/(1 + np.exp(-w.T@x))
        return y_hat

    def fit(self):
        for i in range(self.max_iter):
            wt_new = self.wt + self.alpha*(1/self.N)*self.x@((self.y-self.__logistic(self.x, self.wt)).T)
            self.wt = wt_new
            if self.verbos:
                print('第' + str(i+1) + '次迭代：\n  w=\n', self.wt, '\n')
        
        return 0

    def predict(self, x_new):
        y_hat = self.__logistic(x_new, self.wt)
        return y_hat



if __name__=="__main__":
    x1 = np.sort(-np.random.rand(1,50),1)[::1]
    x2 = np.sort(np.random.rand(1,50),1)[::1]
    x = np.concatenate((x1,x2), axis=1)

    x1 = np.sort(-np.random.rand(1,50),1)[::1]
    x2 = np.sort(np.random.rand(1,50),1)[::1]
    x_new = np.concatenate((x1,x2), axis=1)
    x = np.concatenate((x,x_new), axis=0)

    y = np.concatenate((np.zeros(50),np.ones(50)))[None,:]

    w0 = np.zeros([2,1]) # 初始化参数
    alpha = 0.1
    N = x.shape[0]

    lr = LogisticRegression(x, y, verbos=True)
    lr.fit()
    test_x = np.concatenate((np.linspace(-2,2,100)[None,:], np.linspace(-2,2,100)[None,:]),axis=0)

    test_y_hat = lr.predict(test_x)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(x[0,:].T,x[1,:].T,y.T)
    ax1.plot(x[0,:].squeeze(), x[1,:].squeeze(), test_y_hat.squeeze(), 'orange')
    fig1.show()