import torch
torch.manual_seed(1234)

class LinearModel:

    def __init__(self, w=None):
        self.w = w
        
    def score(self, X):
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))
        return torch.matmul(self.w, X)

    def predict(self, X):
        scores = self.score(X)
        result = torch.zeros(len(scores))

        return 0

class Perceptron(LinearModel):

    def loss(self):
        return 0
    
    def grad(self, X, y):
        return 0

class PerceptronOptimizer:

    def __init__(self, model):
        self.model = model

    def step(self, X, y):
        return 0