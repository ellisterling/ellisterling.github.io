import torch

class LinearModel:
    def __init__(self, w=None):
        self.w = w
        
    def score(self, X):
        if self.w is None: 
            self.w = torch.rand(X.size()[1])
        return torch.matmul(X, self.w)

    def predict(self, X):
        scores = self.score(X)
        return (scores >= 0).float()
    
class Perceptron(LinearModel):
    def loss(self, X, y):
        y_ = 2*y - 1
        scores = self.score(X)
        loss = torch.where(y_*scores <= 0, 1, 0)
        return (1.0*loss).mean()
    
    def grad(self, X, y):
        scores = self.score(X)
        return -((scores*(2*y - 1.0) < 0)*(2*y - 1.0)@X)

class PerceptronOptimizer(Perceptron):
    def step(self, X, y):
        grad = self.grad(X, y)
        self.w -= grad