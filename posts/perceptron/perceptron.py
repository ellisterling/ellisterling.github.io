import torch
torch.manual_seed(1234)

def perceptron_data(n_points = 300, noise = 0.2):
    
    y = torch.arange(n_points) >= int(n_points/2)
    X = y[:, None] + torch.normal(0.0, noise, size = (n_points,2))
    X = torch.cat((X, torch.ones((X.shape[0], 1))), 1)

    return X, y

X, y = perceptron_data(n_points = 300, noise = 0.2)


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
    
p = LinearModel()
s = p.score(X)
result = p.predict(X)
print(result)