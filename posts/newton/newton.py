import torch
torch.manual_seed(1234)

class LinearModel:
    def __init__(self, w=None):
        self.w = w
        
    def score(self, X):
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))
        return torch.matmul(X, self.w)

    def predict(self, X):
        scores = self.score(X)
        return (scores >= 0).float()
    
class LogisticRegression(LinearModel):
    def loss(self, X, y):
        scores = self.score(X)
        probs = self.sigmoid(scores)
        loss = -y * torch.log(probs) - (1 - y) * torch.log(1 - probs)
        return loss.sum()
    
    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))
    
    def hessian(self, X, y):
        # replace  X with xik xij
        scores = self.score(X)
        probs = self.sigmoid(scores)
        hessian = X * probs * (1 - probs)
        return hessian

    def grad(self, X, y):
        scores = self.score(X)
        grad = 0
        for i in range(len(y)):
            grad += (self.sigmoid(scores[i]) - y[i])*X[i]
        return grad/len(y)
        

class GradientDescentOptimizer(LogisticRegression):
    def __init__(self):
        LogisticRegression.__init__(self)
        self.prev_w = self.w

    def step(self, X, y, alpha, beta):
        gradient = self.grad(X, y)
        if self.prev_w is None:
            self.prev_w = self.w.clone()
        temp_w = self.w.clone()
        self.w = self.w - alpha*gradient + beta*(self.w - self.prev_w)
        self.prev_w = temp_w

class NewtonOptimizer(LogisticRegression):
    def __init__(self):
        LogisticRegression.__init__(self)
        self.prev_w = self.w

    def step(self, X, y, alpha):
        gradient = self.grad(X, y)
        hessian = self.hessian(X, y)
        if self.prev_w is None:
            self.prev_w = self.w.clone()
        temp_w = self.w.clone()
        self.w = self.w - alpha * torch.inverse(hessian) * gradient
        self.prev_w = temp_w

class AdamOptimizer(LogisticRegression):
    def __init__(self):
        LogisticRegression.__init__(self)

    def step(self, X, y, batch_size, alpha, beta_1, beta_2, w_0=None):
        if w_0 == None:
            w_0 = torch.rand((X.size()[1]))
            pass
    
