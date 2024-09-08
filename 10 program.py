import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter

    def initialize(self, X):
        self.shape = X.shape
        self.n, self.m = self.shape

        self.phi = np.full(shape=self.k, fill_value=1/self.k)
        self.weights = np.full( shape=self.shape, fill_value=1/self.k)
        
        random_row = np.random.randint(low=0, high=self.n, size=self.k)
        self.mu = [X[row_index,:] for row_index in random_row ]
        self.sigma = [np.cov(X.T) for _ in range(self.k)]

    def e_step(self, X):
        self.weights = self.predict_proba(X)
        self.phi = self.weights.mean(axis=0)

    def m_step(self, X):
        for i in range(self.k):
            weight = self.weights[:, [i]]
            total_weight = weight.sum()
            self.mu[i] = (X * weight).sum(axis=0) / total_weight
            self.sigma[i] = np.cov(X.T, aweights=(weight/total_weight).flatten(), bias=True)

    def fit(self, X):
        self.initialize(X)
        
        for iteration in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)
            
    def predict_proba(self, X):
        likelihood = np.zeros( (self.n, self.k) )
        for i in range(self.k):
            distribution = multivariate_normal(mean=self.mu[i], cov=self.sigma[i])
            likelihood[:,i] = distribution.pdf(X)
        
        numerator = likelihood * self.phi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        return weights
    
    def predict(self, X):
        weights = self.predict_proba(X)
        return np.argmax(weights, axis=1)

np.random.seed(42)
n_samples = 300

X1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_samples // 3)
X2 = np.random.multivariate_normal([4, 4], [[1.5, 0], [0, 1.5]], n_samples // 3)
X3 = np.random.multivariate_normal([-3, 5], [[1.2, 0], [0, 1.2]], n_samples // 3)

X = np.vstack((X1, X2, X3))

gmm = GMM(k=3, max_iter=100)
gmm.fit(X)

labels = gmm.predict(X)

plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('Gaussian Mixture Model Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
