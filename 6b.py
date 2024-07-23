import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train, self.y_train = X, y

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        distances = [manhattan_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return Counter(k_nearest_labels).most_common(1)[0][0]

df = pd.read_csv('/datasetfruit.csv')
label_encoder = LabelEncoder()
df['fruit_name'] = label_encoder.fit_transform(df['fruit_name'])
df['fruit_subtype'] = df['fruit_subtype'].factorize()[0]

X = df.drop('fruit_name', axis=1).values
y = df['fruit_name'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

accuracy = np.mean(predictions == y_test)
print(predictions)
print("Accuracy:", accuracy)
