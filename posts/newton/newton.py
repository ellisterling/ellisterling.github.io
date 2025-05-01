import torch
torch.manual_seed(1234)
import numpy as np

class LinearModel:
    def __init__(self, w=None):
        self.w = w
        
    def score(self, X):
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))
        return torch.mv(X, self.w.double())

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
    
    def hessian(self, X):
        scores = self.score(X)
        probs = self.sigmoid(scores)
        diagonal = torch.diag(probs * (1 - probs))
        return X.T @ diagonal @ X

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
        temp_w = self.w.clone()
        self.w = self.w - alpha*gradient + beta*(self.w - self.prev_w)
        self.prev_w = temp_w

class NewtonOptimizer(LogisticRegression):
    def __init__(self):
        LogisticRegression.__init__(self)
        self.prev_w = self.w

    def step(self, X, y, alpha):
        gradient = self.grad(X, y)
        hessian_matrix = self.hessian(X)
        if self.prev_w is None:
            self.prev_w = self.w.clone()
        temp_w = self.w.clone()
        self.w = self.w - alpha * torch.linalg.solve(hessian_matrix, gradient)
        self.prev_w = temp_w

class AdamOptimizer(LogisticRegression):
    def __init__(self):
        LogisticRegression.__init__(self)

    def step(self, X, y, batch_size, alpha, beta_1, beta_2, w_0=None):
        if w_0 == None:
            w_0 = torch.rand((X.size()[1]))
            pass

'''
Stochastic Gradient Descent

def mse(x, y, w):
    return ((w[1]*x + w[0] - y)**2).mean()

def learning_schedule(t): 
    return 10/(t + 10)

batch_size = 10

# initialize training loop
w = torch.tensor([0.0, 0.0], requires_grad=True)
losses = []
minibatch_losses = []

for t in range(1, 1000):
    # choose a random batch of indices
    i = torch.randint(0, x.shape[0], (batch_size,))

    # compute the loss
    minibatch_loss = mse(x[i], y[i], w)

    # record the minibatchloss
    minibatch_losses.append(minibatch_loss.item())

    # full loss : only for viz, not part of algorithm
    losses.append(mse(x, y, w).item())

    # compute the gradient
    minibatch_loss.backward()

    # update the weights
    # the with statement is boilerplate that tells torch not to keep track of the gradient for the operation of updating w
    with torch.no_grad():
        w -= learning_schedule(t)*w.grad

    # zero the gradient
    w.grad.zero_()
'''
