import torch
torch.manual_seed(1234)

class LinearModel:
    def __init__(self, w=None):
        self.w = w
        
    def score(self, X):
        '''
        Calculates the scores for the current value of w and X.

        Args:
            X: a torch tensor, the feature matrix
        
        Returns:
            a torch tensor, the product of the weight vector w and the feature matrix X
        '''
        if self.w is None: 
            self.w = torch.rand(X.size()[1])
        return torch.matmul(X, self.w)

    def predict(self, X):
        '''
        Calculates binary class predictions for the current value of w and X.

        Args:
            X: a torch tensor, the feature matrix

        Returns:
            a torch tensor of the predicted classes for each item. Items below the w vector are classified as 0,
            and items on or above it are classified as 1.
        '''
        scores = self.score(X)
        return (scores >= 0).float()
    
class LogisticRegression(LinearModel):
    def loss(self, X, y):
        '''
        Calculates the loss for the current state of the weight vector w by comparing the predicted values to the target vector y.

        Args: 
            X: a torch tensor, the feature matrix
            y: a torch tensor, the target vector

        Returns:
            loss: a float value which describes the loss of the model with its current weight vector
        '''
        scores = self.score(X)
        probs = self.sigmoid(scores)
        loss = -y * torch.log(probs) - (1 - y) * torch.log(1 - probs)
        return loss.sum()/len(y)
    
    def sigmoid(self, z):
        '''
        Calculates the sigmoid for a vector z, placing it between 0 and 1.

        Args:
            z: an input vector
        
        Returns:
            a vector of values between 0 and 1
        '''
        return 1 / (1 + torch.exp(-z))
    
    def grad(self, X, y):
        '''
        Calculates the gradient of the feature matrix X. 

        Args:
            X: a torch tensor, the feature matrix
            y: a torch tensor, the target vector
        
        Returns:
            a torch tensor containing the gradient of X
        '''
        scores = self.score(X)
        probs = self.sigmoid(scores)
        grad = (X.T @ (probs - y)) / len(y)
        return grad


class GradientDescentOptimizer(LogisticRegression):
    def __init__(self):
        LogisticRegression.__init__(self)
        self.prev_w = self.w

    def step(self, X, y, alpha, beta):
        '''
        Steps forward the model to optimize the weight vector w. Uses the gradient to approximate the best next value for w.

        Args:
            X: a torch tensor, the feature matrix
            y: a torch tensor, the target vector
            alpha: the learning rate, a float
            beta: the learning rate for gradient descent with momentum, a float between 0 and 1
        
        Returns:
            Nothing--it updates the value stored in self.w
        '''
        gradient = self.grad(X, y)
        if self.prev_w is None:
            self.prev_w = self.w.clone()
        w_next = self.w - alpha*gradient + beta*(self.w - self.prev_w)
        self.prev_w = self.w
        self.w = w_next