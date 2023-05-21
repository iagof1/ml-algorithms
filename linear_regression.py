import numpy as np


class LinearRegression():
  def __init__(self, epochs=3000, lr=0.1):
    self.w, self.b = None, 0
    self.epochs = epochs
    self.lr = lr

  def loss(self, y, y_pred):
    return ((y_pred - y) ** 2 / 2).mean()

  def fit(self, X, y):
    n_samples, n_features = X.shape
    self.w = np.zeros(n_features)

    for i in range(self.epochs):
        y_pred = self.predict(X)
        y_diff = y_pred - y
        dW = (1 / n_samples) * X.T @ y_diff
        db = (1 / n_samples) * np.sum(y_diff)
        self.w -= self.lr * dW
        self.b -= self.lr * db

  def predict(self, X):
    return X @ self.w + self.b
