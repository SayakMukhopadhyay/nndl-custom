import time

import numpy as np


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.rng = np.random.default_rng()

        self.biases = [self.rng.standard_normal(size=(y, 1)) for y in sizes[1:]]
        self.weights = [self.rng.standard_normal(size=(y, x)) for x, y in zip(sizes[:-1], sizes[1:])]

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        test_data_size = None
        if test_data:
            test_data_size = len(test_data)

        start_time = time.time()
        training_data_size = len(training_data)
        for i in range(epochs):
            self.rng.shuffle(training_data)
            mini_batches = [
                training_data[j : j + mini_batch_size] for j in range(0, training_data_size, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.gradient_descent(mini_batch, eta)
                # self.gradient_descent_matrix(mini_batch, eta)

            if test_data:
                print(
                    "Epoch {0}: {1} / {2}, elapsed time: {3:.2f}s".format(
                        i, self.evaluate(test_data), test_data_size, time.time() - start_time
                    )
                )
            else:
                print("Epoch {0} complete, elapsed time: {1:.2f}s".format(i, time.time() - start_time))

    def gradient_descent(self, mini_batch, eta):
        mini_del_bias = [np.zeros(bias.shape) for bias in self.biases]
        mini_del_weight = [np.zeros(weight.shape) for weight in self.weights]

        for first_layer_activations, expected_values in mini_batch:
            del_bias, del_weight = self.back_propagation(first_layer_activations, expected_values)
            mini_del_bias = [mdb + db for mdb, db in zip(mini_del_bias, del_bias)]
            mini_del_weight = [mdw + dw for mdw, dw in zip(mini_del_weight, del_weight)]

        self.biases = [bias - (eta / len(mini_batch)) * mdb for bias, mdb in zip(self.biases, mini_del_bias)]
        self.weights = [weight - (eta / len(mini_batch)) * mdw for weight, mdw in zip(self.weights, mini_del_weight)]

    def gradient_descent_matrix(self, mini_batch, eta):
        mini_batch_first_layer, mini_batch_expected_value = zip(*mini_batch)

        del_bias, del_weight = self.back_propagation(
            np.squeeze(np.array(mini_batch_first_layer)).transpose(),
            np.squeeze(np.array(mini_batch_expected_value)).transpose(),
        )

        self.biases = [bias - (eta / len(mini_batch)) * mdb for bias, mdb in zip(self.biases, del_bias)]
        self.weights = [weight - (eta / len(mini_batch)) * mdw for weight, mdw in zip(self.weights, del_weight)]

    def back_propagation(self, first_layer_activations, expected_values):
        activations, z_vectors = self.feed_forward(first_layer_activations)

        del_bias = [np.zeros(bias.shape) for bias in self.biases]
        del_weight = [np.zeros(weight.shape) for weight in self.weights]

        # This calculates (a - y) * (d(sigmoid(z)/dz) which is dC/dz
        # In every step, the delta multiplied by the activation of the previous layer gives the partial gradient
        delta = self.cost_derivative(activations[-1], expected_values) * sigmoid_derivative(z_vectors[-1])
        del_bias[-1] = np.sum(delta, axis=1, keepdims=True)
        # The transpose is cause both the delta and activations a column matrix
        del_weight[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_derivative(z_vectors[-l])
            del_bias[-l] = np.sum(delta, axis=1, keepdims=True)
            del_weight[-l] = np.dot(delta, activations[-l - 1].transpose())

        return del_bias, del_weight

    # This function calculates the z = w.a + b and sigmoids it for every layer
    def feed_forward(self, activation):
        activations = [activation]
        z_vectors = []
        for b, w in zip(self.biases, self.weights):
            z_vector = np.dot(w, activation) + b
            z_vectors.append(z_vector)
            activation = sigmoid(z_vector)
            activations.append(activation)
        return activations, z_vectors

    # This provides the derivative of the cost function (a - y)^2 w.r.t a which results in just (a - y)
    def cost_derivative(self, output_layer_activations, expected_values):
        return output_layer_activations - expected_values

    def evaluate(self, test_data):
        # Based on the trained weights and biases, the test data is used to predict the output activations
        # self.feed_forward(x) gives a tuple of the activations and z vectors of all layers
        test_results = [(np.argmax(self.feed_forward(x)[0][-1]), y) for x, y in test_data]
        return [sum(int(prediction == actual) for prediction, actual in test_results)]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    sigmoid_value = sigmoid(z)
    return sigmoid_value * (1 - sigmoid_value)
