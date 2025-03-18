import numpy as np

def sigmoid(x: float) -> float: # Sigmoid as our activation function
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x: float) -> float: # Derivative of Sigmoid
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true: float, y_pred: float) -> float: # mean squared error loss
    return ((y_true - y_pred) ** 2).mean()

class NeuralNetwork:
    def __init__(self, layer_sizes) -> None:
        self.layer_sizes = layer_sizes # List of integers representing the number of neurons in each layer
        self.weights = [] # Weights
        self.biases = [] # Biases
        # Initialize weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01 # small random weights for start
            bias = np.zeros((1, layer_sizes[i + 1])) # zero bias for start
            self.weights.append(weight)
            self.biases.append(bias)
    
    def feedforward(self , x: float) -> float:
        # Compute the forward pass for a single input x
        activations = [np.array(x).reshape(1, -1)] # Start with input as a row vector
        for weight, bias in zip(self.weights, self.biases):
            weighted_sum = np.dot(activations[-1], weight) + bias # weighted sum + bias
            activ_func = sigmoid(weighted_sum) # Apply activation function
            activations.append(activ_func)
        return activations[-1] # Return the output layer's activation
    
    def train(self, data: float, all_y_trues: float, learn_rate: float = 0.1, epochs: int = 30000) -> None:

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # Forward pass
                x = np.array(x).reshape(1, -1)
                activations = [x]
                sums = [] # Store pre-activation sums for backprop
                for weight, bias in zip(self.weights, self.biases):
                    z = np.dot(activations[-1], weight) + bias
                    sums.append(z)
                    a = sigmoid(z)
                    activations.append(a)
                y_pred = activations[-1]

                 # Backward pass
                delta = -2 * (y_true - y_pred) * deriv_sigmoid(sums[-1])  # Gradient at output
                deltas = [delta]

                # Compute gradients for hidden layers
                for i in range(len(self.weights) - 2, -1, -1):
                    delta = np.dot(deltas[0], self.weights[i + 1].T) * deriv_sigmoid(sums[i])
                    deltas.insert(0, delta)

                # Update weights and biases
                for i in range(len(self.weights)):
                    self.weights[i] -= learn_rate * np.dot(activations[i].T, deltas[i])
                    self.biases[i] -= learn_rate * deltas[i]

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data).flatten()
                loss = mse_loss(all_y_trues, y_preds)
                print(f'Epoch {epoch} loss: {loss}')

    def predict(self, x: float) -> float:
        #Run a single test input through the network.
        pred = self.feedforward(x)[0][0]
        return pred

if __name__ == "__main__":
    # Example usage with XOR
    data = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
        ])
    all_y_trues = np.array([
        0,
        1,
        1,
        0
        ])
    # Define a network with 2 input neurons, 4 hidden neurons, 1 output neuron
    network = NeuralNetwork([2, 4, 1])
    # Train the network
    network.train(data, all_y_trues)
    for x in data:
        pred = network.predict(x)
        print(f'Input: {x}, Prediction: {pred}')