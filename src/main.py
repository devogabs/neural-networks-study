import numpy as np

# Define the sigmoid activation function
def sigmoid(x, derivative=False):
    if derivative==True:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# Example input dataset

X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

# Example output dataset
y = np.array([[0,0,1,1]]).T # transpose to make it a column vector
# Seed random numbers to make calculation deterministic (just for this example)
np.random.seed(1) # this number 1 makes the random numbers the same every time

# Initialize weights randomly with mean 0
syn0 = 2 * np.random.random((3, 1)) - 1 # 3 input nodes, 1 output node. This creates a 3x1 matrix of weights

for iter in range(10000):

    # Forward propagation
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, syn0))

    # Calculate the error
    l1_error = y - layer_1

    # Multiply the error by the derivative of the sigmoid function
    l1_delta = l1_error * sigmoid(layer_1, derivative=True)

    # Update weights
    syn0 += np.dot(layer_0.T, l1_delta)

print("Output after training:")
print(layer_1)
# The output should be close to the expected output [0, 0, 1, 1]

