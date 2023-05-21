import numpy as np

class LogisticRegression():
  def __init__(self, epochs=1000, lr=0.01):
    self.w, self.b = None, 0
    self.epochs = epochs
    self.lr = lr

  def loss(self, y, y_pred):
    return ((y_pred - y) ** 2 / 2).mean()

  def sigmoid(self, X):
    return 1/(1+np.exp(-X))

  def fit(self, X, y):
    n_samples, n_features = X.shape
    self.w = np.zeros(n_features)

    for i in range(self.epochs):
        y_pred = self._predict(X)
        y_diff = y_pred - y

        dW = (1 / n_samples) * X.T @ y_diff
        db = (1 / n_samples) * np.sum(y_diff)
        self.w -= self.lr * dW
        self.b -= self.lr * db

  def _predict(self, X):
    return self.sigmoid(X @ self.w + self.b)

  def predict(self, X):
    return [0 if y_pred <= 0.5 else 1 for y_pred in self._predict(X)]