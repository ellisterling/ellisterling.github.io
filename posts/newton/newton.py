import torch
torch.manual_seed(1234)
import numpy as np

class LinearModel:
    def __init__(self):
        self.w = None
        
    def score(self, X):
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))
        return torch.matmul(X, self.w.double())

    def predict(self, X):
        scores = self.score(X)
        return (scores >= 0).float()
    
class LogisticRegression(LinearModel):
    def loss(self, X, y):
        scores = self.score(X)
        probs = self.sigmoid(scores)
        loss = -y * torch.log(probs) - (1 - y) * torch.log((1 - probs))
        return loss.sum()
    
    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))
    
    def grad(self, X, y):
        scores = self.score(X)
        probs = self.sigmoid(scores)
        return X.T @ (probs - y) / len(y)
        

class GradientDescentOptimizer(LogisticRegression):
    def __init__(self):
        LogisticRegression.__init__(self)
        self.prev_w = self.w

    def step(self, X, y, alpha, beta):
        gradient = self.grad(X, y)
        if self.prev_w is None:
            self.prev_w = self.w.clone()
        w_next = self.w - alpha*gradient + beta*(self.w - self.prev_w)
        self.prev_w = self.w
        self.w = w_next

class NewtonOptimizer(LogisticRegression):
    def hessian(self, X, y):
        scores = self.score(X)
        probs = self.sigmoid(scores)
        diagonal = torch.diag(probs * (1 - probs))
        return (X.T @ diagonal @ X)/len(y)
    
    def step(self, X, y, alpha):
        gradient = self.grad(X, y)
        hessian_matrix = self.hessian(X, y)
        self.w = self.w - alpha * torch.pinverse(hessian_matrix) @ gradient

class AdamOptimizer(LogisticRegression):
    def __init__(self):
        LogisticRegression.__init__(self)
        self.m = 0
        self.prev_m = 0
        self.v = 0
        self.prev_v = 0

    def step(self, X, y, batch_size, alpha, beta_1, beta_2, t, w_0=None):
        if w_0 == None:
            w_0 = torch.rand((X.size()[1]))
        
        # take random selection of {batch_size} indices from x.
        i = torch.randint(0, len(y), (batch_size,))
        grad = self.grad(X[i], y[i])

        # get first and second moment estimates
        m_next = beta_1*self.prev_m + (1 - beta_1)*grad
        v_next = beta_2*self.prev_v + (1 - beta_2)*grad*grad
        self.prev_m = self.m
        self.m = m_next
        self.prev_v = self.v
        self.v = v_next

        # adjust m and v for possible bias -- because they start at 0 we don't want them to be biased towards that
        e = 1*10**(-8) # to make sure no division by 0 happens
        m_hat = self.m/(1 - beta_1**t)
        v_hat = self.v/(1 - beta_2**t)

        # Update w
        self.w = self.w - alpha * (m_hat/(torch.sqrt(v_hat)+e))