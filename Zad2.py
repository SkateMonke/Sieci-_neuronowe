""""
Do uzycia innej funkcji niz ReLU nalezy zmienic funkcje aktywacji w forward prop, back prop i testach
oraz zakomentowac skalowanie danych uczacych, oraz w tych w czasie procesu uczenia
"""

import numpy as np
import matplotlib.pyplot as plt


def ReLU(Z):
    return np.maximum(Z, 0)

def Sigmoid(Z):
    return 1/(1 + np.exp(-Z*0.5))

def ReLU_deriv(Z):
    return Z > 0

def Sigmoid_deriv(Z):
    return Sigmoid(Z) * (1 - Sigmoid(Z))

x_data = np.array([0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600], dtype="float64")
y_data = np.array([1.0, 1.32, 1.6, 1.54, 1.41, 1.01, 0.6, 0.42, 0.2, 0.51, 0.8], dtype="float64")

skala_x = np.max(x_data)
skala_y = np.max(y_data)

# x_data /= skala_x
# y_data /= skala_y

input_size = 1
hidden_size_1 = 5
hidden_size_2 = 5
output_size = 1

weights_input_hidden1 = np.random.rand(input_size, hidden_size_1) * 0.1
weights_hidden1_hidden2 = np.random.rand(hidden_size_1, hidden_size_2) * 0.1
weights_hidden2_output = np.random.rand(hidden_size_2, output_size) * 0.1
bias_hidden1 = np.zeros((1, hidden_size_1))
bias_hidden2 = np.zeros((1, hidden_size_2))
bias_output = np.zeros((1, output_size))

learning_rate = 0.0001
epochs = 1000000

for epoch in range(epochs):
    #Forward prop
    hidden1_input = np.dot(x_data.reshape(-1, 1), weights_input_hidden1) + bias_hidden1
    hidden1_output = ReLU(hidden1_input)

    hidden2_input = np.dot(hidden1_output, weights_hidden1_hidden2) + bias_hidden2
    hidden2_output = ReLU(hidden2_input)

    output_layer_input = np.dot(hidden2_output, weights_hidden2_output) + bias_output
    #predicted_output = Sigmoid(output_layer_input)
    predicted_output = ReLU(output_layer_input)

    #MSE
    MSE = np.mean((predicted_output - y_data.reshape(-1, 1)) ** 2)

    #Back prop
    error_output = y_data.reshape(-1, 1) - predicted_output
    delta_output = error_output * ReLU_deriv(predicted_output)

    error_hidden2 = delta_output.dot(weights_hidden2_output.T)
    delta_hidden2 = error_hidden2 * ReLU_deriv(hidden2_output)

    error_hidden1 = delta_hidden2.dot(weights_hidden1_hidden2.T)
    delta_hidden1 = error_hidden1 * ReLU_deriv(hidden1_output)

    #Update
    weights_hidden2_output += hidden2_output.T.dot(delta_output) * learning_rate
    weights_hidden1_hidden2 += hidden1_output.T.dot(delta_hidden2) * learning_rate
    weights_input_hidden1 += x_data.reshape(-1, 1).T.dot(delta_hidden1) * learning_rate

    bias_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
    bias_hidden2 += np.sum(delta_hidden2, axis=0, keepdims=True) * learning_rate
    bias_hidden1 += np.sum(delta_hidden1, axis=0, keepdims=True) * learning_rate

    if epoch % 5000 == 0:
        print(f"Epoch {epoch}, Error: {MSE}")

#Test
test_input = np.linspace(min(x_data), max(x_data), 110).reshape(-1, 1)
test_hidden1 = ReLU(np.dot(test_input, weights_input_hidden1) + bias_hidden1)
test_hidden2 = ReLU(np.dot(test_hidden1, weights_hidden1_hidden2) + bias_hidden2)
test_output = ReLU(np.dot(test_hidden2, weights_hidden2_output) + bias_output)

#plt.scatter(x_data * skala_x, y_data * skala_y)
#plt.plot(test_input * skala_x, test_output * skala_y, c="r")
plt.scatter(x_data, y_data)
plt.plot(test_input, test_output, c="r")
#plt.scatter(test_input * skala_x, test_output * skala_y)
plt.show()
