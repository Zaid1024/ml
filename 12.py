import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.losses import BinaryCrossentropy

# AND-NOT function
def and_not(x):
    return np.logical_and(x[:, 0], np.logical_not(x[:, 1])).astype(int)

# XOR function
def xor(x):
    return np.logical_xor(x[:, 0], x[:, 1]).astype(int)

# Training data for AND-NOT
x_train_and_not = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train_and_not = and_not(x_train_and_not)

# Training data for XOR
x_train_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train_xor = xor(x_train_xor)

# Build the model for AND-NOT
model_and_not = Sequential()
model_and_not.add(Dense(4, input_dim=2, activation='relu'))
model_and_not.add(Dense(1, activation='sigmoid'))
model_and_not.compile(optimizer=SGD(learning_rate=0.1), loss=BinaryCrossentropy(), metrics=['accuracy'])

# Build the model for XOR
model_xor = Sequential()
model_xor.add(Dense(4, input_dim=2, activation='relu'))
model_xor.add(Dense(1, activation='sigmoid'))
model_xor.compile(optimizer=SGD(learning_rate=0.1), loss=BinaryCrossentropy(), metrics=['accuracy'])

# Train the model for AND-NOT
model_and_not.fit(x_train_and_not, y_train_and_not, epochs=1000, verbose=0)

# Train the model for XOR
model_xor.fit(x_train_xor, y_train_xor, epochs=1000, verbose=0)

# Evaluate the model for AND-NOT
print("AND-NOT Model Evaluation:")
and_not_predictions = model_and_not.predict(x_train_and_not)
print(and_not_predictions.round())
print("Expected Output:")
print(y_train_and_not)

# Evaluate the model for XOR
print("XOR Model Evaluation:")
xor_predictions = model_xor.predict(x_train_xor)
print(xor_predictions.round())
print("Expected Output:")
print(y_train_xor)
