import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, y):
        self.X_train, self.y_train = X, y

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        if self.distance_metric == 'euclidean':
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.distance_metric == 'manhattan':
            distances = [manhattan_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return Counter(k_nearest_labels).most_common(1)[0][0]

# Load the Glass dataset
df = pd.read_csv('./dataset/glass.csv')
X = df.drop('Type', axis=1).values
y = df['Type'].values

# Split the dataset into training and testing sets (70-30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and fit the KNN model with Euclidean distance
knn_euclidean = KNN(k=3, distance_metric='euclidean')
knn_euclidean.fit(X_train, y_train)
predictions_euclidean = knn_euclidean.predict(X_test)
accuracy_euclidean = np.mean(predictions_euclidean == y_test)
print("Euclidean Distance - Predictions:", predictions_euclidean)
print("Euclidean Distance - Accuracy:", accuracy_euclidean)

# Initialize and fit the KNN model with Manhattan distance
knn_manhattan = KNN(k=3, distance_metric='manhattan')
knn_manhattan.fit(X_train, y_train)
predictions_manhattan = knn_manhattan.predict(X_test)
accuracy_manhattan = np.mean(predictions_manhattan == y_test)
print("Manhattan Distance - Predictions:", predictions_manhattan)
print("Manhattan Distance - Accuracy:", accuracy_manhattan)