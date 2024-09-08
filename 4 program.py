import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.1):
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def forward(self, X):
        self.activations = [X]
        self.zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, self.activations[-1]) + b
            self.zs.append(z)
            self.activations.append(self.sigmoid(z))
        return self.activations[-1]

    def backward(self, X, y):
        m = X.shape[1]
        delta = (self.activations[-1] - y) * self.sigmoid_derivative(self.activations[-1])
        
        nabla_w = [np.dot(delta, self.activations[-2].T)]
        nabla_b = [np.sum(delta, axis=1, keepdims=True)]
        
        for l in range(2, len(self.layers)):
            delta = np.dot(self.weights[-l+1].T, delta) * self.sigmoid_derivative(self.activations[-l])
            nabla_w.insert(0, np.dot(delta, self.activations[-l-1].T))
            nabla_b.insert(0, np.sum(delta, axis=1, keepdims=True))
        
        self.weights = [w - self.learning_rate * nw / m for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - self.learning_rate * nb / m for b, nb in zip(self.biases, nabla_b)]

    def train(self, X, y, epochs):
        for _ in range(epochs):
            self.forward(X)
            self.backward(X, y)

    def predict(self, X):
        return self.forward(X)

X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
y = np.array([[0, 1, 1, 0]])

nn = NeuralNetwork([2, 4, 1], learning_rate=0.1)
nn.train(X, y, epochs=10000)

test_data = np.array([[1, 1, 0, 0],
                      [1, 0, 1, 0]])
predictions = nn.predict(test_data)

print("Test Data:")
print(test_data.T)
print("\nPredictions:")
print(predictions.T)
