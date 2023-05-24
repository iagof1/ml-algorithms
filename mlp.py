import numpy as np

class MLP:
    def __init__(self, epochs=5000, lr=0.1):
        self.epochs = epochs
        self.W0 = np.random.random((2, 2))
        self.W1 = np.random.random((2, 2))
        self.W2 = np.random.random((2, 1))
        self.lr = lr

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def sigmoid_derivative(self, X):
        return X * (1 - X)

    def fit(self, X, y):
        for epoch in range(self.epochs):
            # forward propagation
            L1 = self.sigmoid(np.dot(X, self.W0))
            L2 = self.sigmoid(np.dot(L1, self.W1))
            L3 = self.sigmoid(np.dot(L2, self.W2))

            # back propagate
            L3_error = y - L3
            L3_delta = L3_error * self.sigmoid_derivative(L3)

            L2_error = np.dot(L3_delta, self.W2.T)
            L2_delta = L2_error * self.sigmoid_derivative(L2)

            L1_error = np.dot(L2_delta, self.W1.T)
            L1_delta = L1_error * self.sigmoid_derivative(L1)

            self.W2 += self.lr * np.dot(L2.T, L3_delta)
            self.W1 += self.lr * np.dot(L1.T, L2_delta)
            self.W0 += self.lr * np.dot(X.T, L1_delta)

    def predict(self, X):
        L1 = self.sigmoid(np.dot(X, self.W0))
        L2 = self.sigmoid(np.dot(L1, self.W1))
        L3 = self.sigmoid(np.dot(L2, self.W2))
        return (L3 > 0.5).astype(int)

X_train = np.array([
    [0, 0],
    [1, 1],
    [1, 0],
    [1, 1],
    [0, 0],
    [1, 1],
    [1, 1],
    [1, 1],
    [0, 1],
    [0, 1],
    [1, 0],
    [1, 1],
])

X_test = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])


# AND gate
y = np.array([
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
    [1],
    [1],
    [0],
    [0],
    [0],
    [1]

])

mlp = MLP()
mlp.fit(X_train, y)
predicted = mlp.predict(X_test)

print(predicted)
