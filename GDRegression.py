
import random

class MySGDRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    # simple stochastic GD
    def fit(self, x, y, learningRate = 0.001, noEpochs = 1000):
        self.coef_ = [0.0 for _ in range(len(x[0]) + 1)]    #beta or w coefficients y = w0 + w1 * x1 + w2 * x2 + ...
        # self.coef_ = [random.random() for _ in range(len(x[0]) + 1)]    #beta or w coefficients
        for epoch in range(noEpochs):
            # TBA: shuffle the trainind examples in order to prevent cycles
            for i in range(len(x)): # for each sample from the training data
                ycomputed = self.eval(x[i])     # estimate the output
                crtError = ycomputed - y[i]     # compute the error for the current sample
                for j in range(0, len(x[0])):   # update the coefficients
                    self.coef_[j] = self.coef_[j] - learningRate * crtError * x[i][j]
                self.coef_[len(x[0])] = self.coef_[len(x[0])] - learningRate * crtError * 1

        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]

    def eval(self, xi):
        yi = self.coef_[-1]
        for j in range(len(xi)):
            yi += self.coef_[j] * xi[j]
        return yi

    def predict(self, x):
        yComputed = [self.eval(xi) for xi in x]
        return yComputed

class MyBatchGDRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    def fit(self, x, y, learningRate=0.001, noEpochs=1000, batchSize=10):
        self.coef_ = [0.0 for _ in range(len(x[0]) + 1)]
        num_batches = len(x) // batchSize
        for epoch in range(noEpochs):
            # Shuffle the training examples for each epoch
            # x, y = random.shuffle(x, y)
            for batch_idx in range(num_batches):
                batch_x = x[batch_idx*batchSize:(batch_idx+1)*batchSize]
                batch_y = y[batch_idx*batchSize:(batch_idx+1)*batchSize]
                batch_size = len(batch_x)
                batch_grad = [0.0 for _ in range(len(x[0]) + 1)]
                for i in range(batch_size):
                    ycomputed = self.eval(batch_x[i])
                    crtError = ycomputed - batch_y[i]
                    for j in range(len(batch_x[i])):
                        batch_grad[j] += crtError * batch_x[i][j]
                    batch_grad[-1] += crtError * 1
                for j in range(len(x[0]) + 1):
                    self.coef_[j] -= learningRate * batch_grad[j] / batch_size

        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]

    def eval(self, xi):
        yi = self.coef_[-1]
        for j in range(len(xi)):
            yi += self.coef_[j] * xi[j]
        return yi

    def predict(self, x):
        yComputed = [self.eval(xi) for xi in x]
        return yComputed
