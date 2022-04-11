import numpy as np
import pickle

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x>0)*1

class Network:
    
    def __init__(self, X, y, learning_rate, reg_lambda, hdim):
        
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.hdim = hdim
        self.indim = 784
        self.outdim = 10
        np.random.seed(13)
        self.W1 = np.random.uniform(-0.1, 0.1, (self.indim, hdim))
        self.b1 = np.random.uniform(-0.1,0.1,(1, hdim))
        self.W2 = np.random.uniform(-0.1, 0.1, (hdim, self.outdim))
        self.b2 = np.random.uniform(-0.1,0.1,(1, self.outdim))

    def forward(self, X):

        z1 = X.dot(self.W1) + self.b1
        a1 = relu(z1)

        z2 = a1.dot(self.W2) + self.b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def lost_func(self,X,y):
        n = X.shape[0]
        y_pred = self.forward(X)
        ls = 0
        for i in range(n):
            ls += -np.log(y_pred[i,y[i,0]])
        ls += self.reg_lambda* (np.sum(self.W1**2) + np.sum(self.W2**2))/2
        return ls

    def sgd_step(self,x,y):

        n = x.shape[0]
        np.random.seed(19)
        i = np.random.randint(n)
        x0 = x[i,:].reshape(1,self.indim)
        y0 = y[i,0]

        z1 = x0.dot(self.W1)+self.b1
        a1 = relu(z1)

        z2 = a1.dot(self.W2)+self.b2
        e = np.exp(z2)
        sume = np.sum(e)
        e /= sume

        delta = e.copy()
        delta[0,y0] -= 1

        dw2 = a1.T.dot(delta)
        db2 = np.sum(delta,axis=0,keepdims=True)


        delta = (delta.dot(self.W2.T)) * ((a1>0).reshape(a1.shape))

        dw1 = x0.T.dot(delta)
        db1 = np.sum(delta,axis=0,keepdims=True)

        dw1 += self.reg_lambda*self.W1/n
        dw2 += self.reg_lambda*self.W2/n

        self.W1 -= self.learning_rate*dw1/n
        self.b1 -= self.learning_rate*db1/n
        self.W2 -= self.learning_rate*dw2/n
        self.b2 -= self.learning_rate*db2/n


    def predict(self, X_in, y_in):
        y_pred = np.argmax(self.forward(X_in),axis=1)
        y_pred = y_pred.reshape(y_pred.shape[0],1)

        accuracy = np.sum(y_pred-y_in==0) / y_in.shape[0] 

        return accuracy


    def save(self,fname='model.pickle'):
        f = open(fname,'wb')
        pickle.dump(self,f)
        f.close()