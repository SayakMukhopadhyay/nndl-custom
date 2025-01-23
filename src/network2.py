import time

import numpy as np


class QuadraticCost(object):
    @staticmethod
    def cost(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoid_derivative(z)


class CrossEntropyCost(object):
    @staticmethod
    def cost(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        return a - y


class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.rng = np.random.default_rng()

        self.msr_weight_initializer()

        self.cost = cost

    def msr_weight_initializer(self):
        self.biases = [self.rng.standard_normal(size=(y, 1)) for y in self.sizes[1:]]
        self.weights = [
            self.rng.standard_normal(size=(y, x)) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def large_weight_initializer(self):
        self.biases = [self.rng.standard_normal(size=(y, 1)) for y in self.sizes[1:]]
        self.weights = [self.rng.standard_normal(size=(y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def stochastic_gradient_descent(
        self,
        training_data,
        epochs,
        mini_batch_size,
        initial_eta,
        lmbda=0.0,
        early_stopping=None,
        learning_rate_schedule=None,
        evaluation_data=None,
        monitor_evaluation_cost=False,
        monitor_evaluation_accuracy=False,
        monitor_training_cost=False,
        monitor_training_accuracy=False,
    ):
        evaluation_data_size = None
        if evaluation_data:
            evaluation_data_size = len(evaluation_data)

        training_data_size = len(training_data)

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        start_time = time.time()

        i = 0
        eta = initial_eta

        while True:
            self.rng.shuffle(training_data)
            mini_batches = [
                training_data[j : j + mini_batch_size] for j in range(0, training_data_size, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                # self.gradient_descent(mini_batch, eta, lmbda, training_data_size)
                self.gradient_descent_matrix(mini_batch, eta, lmbda, training_data_size)

            print("Epoch {0} training complete, elapsed time: {1:.2f}s".format(i, time.time() - start_time))
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {0}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {0} / {1}".format(accuracy, training_data_size))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {0}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {0} / {1}".format(accuracy, evaluation_data_size))
            print()

            i = i + 1

            if early_stopping is None:
                if i < epochs:
                    print("Max epochs {0} reached. Stopping...".format(epochs))
                    break
            else:
                if (
                    len(evaluation_accuracy) > early_stopping
                    and max(evaluation_accuracy[-early_stopping:]) < evaluation_accuracy[-early_stopping - 1]
                ):
                    if learning_rate_schedule is None:
                        print("No improvement in evaluation accuracy in {0} epochs. Stopping...".format(early_stopping))
                        break
                    else:
                        print(
                            "No improvement in evaluation accuracy in {0} epochs. Halving learning rate...".format(
                                early_stopping
                            )
                        )
                        eta = eta / 2
                        if initial_eta / learning_rate_schedule >= eta:
                            print(
                                "Learning rate halved to {0} of initial value. Stopping...".format(
                                    learning_rate_schedule
                                )
                            )
                            break

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def gradient_descent(self, mini_batch, eta, lmbda, training_data_size):
        mini_del_bias = [np.zeros(bias.shape) for bias in self.biases]
        mini_del_weight = [np.zeros(weight.shape) for weight in self.weights]

        for first_layer_activations, expected_values in mini_batch:
            del_bias, del_weight = self.back_propagation(first_layer_activations, expected_values)
            mini_del_bias = [mdb + db for mdb, db in zip(mini_del_bias, del_bias)]
            mini_del_weight = [mdw + dw for mdw, dw in zip(mini_del_weight, del_weight)]

        self.biases = [bias - (eta / len(mini_batch)) * mdb for bias, mdb in zip(self.biases, mini_del_bias)]
        self.weights = [
            (1 - eta * (lmbda / training_data_size)) * weight - (eta / len(mini_batch)) * mdw
            for weight, mdw in zip(self.weights, mini_del_weight)
        ]

    def gradient_descent_matrix(self, mini_batch, eta, lmbda, training_data_size):
        mini_batch_first_layer, mini_batch_expected_value = zip(*mini_batch)

        del_bias, del_weight = self.back_propagation(
            np.squeeze(np.array(mini_batch_first_layer)).transpose(),
            np.squeeze(np.array(mini_batch_expected_value)).transpose(),
        )

        self.biases = [bias - (eta / len(mini_batch)) * mdb for bias, mdb in zip(self.biases, del_bias)]
        self.weights = [
            (1 - eta * (lmbda / training_data_size)) * weight - (eta / len(mini_batch)) * mdw
            for weight, mdw in zip(self.weights, del_weight)
        ]

    def back_propagation(self, first_layer_activations, expected_values):
        activations, z_vectors = self.feed_forward(first_layer_activations)

        del_bias = [np.zeros(bias.shape) for bias in self.biases]
        del_weight = [np.zeros(weight.shape) for weight in self.weights]

        # This calculates (a - y) * (d(sigmoid(z)/dz) which is dC/dz
        # In every step, the delta multiplied by the activation of the previous layer gives the partial gradient
        delta = self.cost.delta(z_vectors[-1], activations[-1], expected_values)
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

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        data_size = len(data)
        for first_layer_activations, expected_values in data:
            activations, z_vectors = self.feed_forward(first_layer_activations)
            if convert:
                expected_values = vectorized_result(expected_values)

            cost += self.cost.cost(activations[-1], expected_values) / data_size

        cost += 0.5 * (lmbda / data_size) * sum(np.linalg.norm(weight) ** 2 for weight in self.weights)
        return cost

    def accuracy(self, data, convert=False):
        if convert:
            results = [
                (np.argmax(self.feed_forward(first_layer_activations)[0][-1]), np.argmax(expected_values))
                for (first_layer_activations, expected_values) in data
            ]
        else:
            results = [
                (np.argmax(self.feed_forward(first_layer_activations)[0][-1]), expected_values)
                for (first_layer_activations, expected_values) in data
            ]
        return [sum(int(prediction == actual) for prediction, actual in results)]


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    sigmoid_value = sigmoid(z)
    return sigmoid_value * (1 - sigmoid_value)
