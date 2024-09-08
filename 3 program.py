import pandas as pd
import numpy as np
from collections import Counter

class Node:
    def __init__(self, attribute=None, label=None, branches=None):
        self.attribute = attribute
        self.label = label
        self.branches = branches or {}

def entropy(y):
    counts = Counter(y)
    probabilities = [count / len(y) for count in counts.values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

def information_gain(X, y, attribute):
    total_entropy = entropy(y)
    weighted_entropy = 0
    for value in X[attribute].unique():
        subset = y[X[attribute] == value]
        weighted_entropy += len(subset) / len(y) * entropy(subset)
    return total_entropy - weighted_entropy

def id3(X, y, attributes):
    if len(set(y)) == 1:
        return Node(label=y.iloc[0])
    
    if len(attributes) == 0:
        return Node(label=Counter(y).most_common(1)[0][0])
    
    gains = {attr: information_gain(X, y, attr) for attr in attributes}
    best_attribute = max(gains, key=gains.get)
    
    node = Node(attribute=best_attribute)
    
    for value in X[best_attribute].unique():
        subset_X = X[X[best_attribute] == value].drop(best_attribute, axis=1)
        subset_y = y[X[best_attribute] == value]
        new_attributes = [attr for attr in attributes if attr != best_attribute]
        
        if len(subset_y) == 0:
            node.branches[value] = Node(label=Counter(y).most_common(1)[0][0])
        else:
            node.branches[value] = id3(subset_X, subset_y, new_attributes)
    
    return node

def predict(node, instance):
    if node.label is not None:
        return node.label
    value = instance[node.attribute]
    if value not in node.branches:
        return max(Counter(node.branches.values()).items(), key=lambda x: x[1])[0]
    return predict(node.branches[value], instance)

def print_tree(node, indent=""):
    if node.label is not None:
        print(f"{indent}Leaf: {node.label}")
    else:
        print(f"{indent}{node.attribute}")
        for value, child in node.branches.items():
            print(f"{indent}  {value} ->")
            print_tree(child, indent + "    ")

data = pd.read_csv(r"C:\Users\Welcome\OneDrive\Documents\decisiontree.csv")
X = data.drop('Play', axis=1)
y = data['Play']

attributes = list(X.columns)
tree = id3(X, y, attributes)

print("Decision Tree:")
print_tree(tree)

new_sample = pd.Series({
    'Outlook': 'Sunny',
    'Temperature': 'Cool',
    'Humidity': 'High',
    'Wind': 'Strong'
})

prediction = predict(tree, new_sample)
print("\nNew sample:", new_sample.to_dict())
print("Prediction:", prediction)
