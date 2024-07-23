import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Define the dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
y_or = np.array([0, 1, 1, 1])

# Define the model for AND function
model_and = Sequential()
model_and.add(Dense(1, input_dim=2, activation='sigmoid'))
model_and.compile(optimizer=SGD(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model for AND function
model_and.fit(X, y_and, epochs=1000, verbose=0)

# Define the model for OR function
model_or = Sequential()
model_or.add(Dense(1, input_dim=2, activation='sigmoid'))
model_or.compile(optimizer=SGD(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model for OR function
model_or.fit(X, y_or, epochs=1000, verbose=0)

# Evaluate the AND function
print("AND function:")
for x in X:
    y_and_pred = model_and.predict(np.array([x]), verbose=0)
    print(f"Input: {x}, Output: {1 if y_and_pred > 0.5 else 0}")

# Evaluate the OR function
print("\nOR function:")
for x in X:
    y_or_pred = model_or.predict(np.array([x]), verbose=0)
    print(f"Input: {x}, Output: {1 if y_or_pred > 0.5 else 0}")
