import numpy as np
import pytensor
import pytensor.tensor as tensor
from pytensor.tensor.math import sigmoid
from pytensor.tensor.random.utils import RandomStream
from pytensor.tensor.special import softmax

pytensor.config.floatX = "float32"


class FullyConnectedLayer(object):
    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout

        self.rng = np.random.default_rng()

        self.w = pytensor.shared(
            np.asarray(
                self.rng.normal(loc=0, scale=np.sqrt(1 / n_out), size=(n_in, n_out)), dtype=pytensor.config.floatX
            ),
            name="w",
            borrow=True,
        )
        self.b = pytensor.shared(
            np.asarray(self.rng.standard_normal(size=(n_out,)), dtype=pytensor.config.floatX), name="b", borrow=True
        )

        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn((1 - self.p_dropout) * tensor.dot(self.inpt, self.w) + self.b)

        self.y_out = tensor.argmax(self.output, axis=1)

        self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(tensor.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        return tensor.mean(tensor.eq(y, self.y_out))


class SoftmaxLayer(object):
    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout

        self.rng = np.random.default_rng()

        self.w = pytensor.shared(
            np.zeros((n_in, n_out), dtype=pytensor.config.floatX),
            name="w",
            borrow=True,
        )
        self.b = pytensor.shared(
            np.zeros((n_out,), dtype=pytensor.config.floatX),
            name="b",
            borrow=True,
        )

        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1 - self.p_dropout) * tensor.dot(self.inpt, self.w) + self.b)

        self.y_out = tensor.argmax(self.output, axis=1)

        self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(tensor.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        return -tensor.mean(tensor.log(self.output_dropout)[tensor.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        return tensor.mean(tensor.eq(y, self.y_out))


class Network(object):
    def __init__(self, layers, mini_batch_size, training_data):
        self.layers = layers
        self.mini_batch_size = mini_batch_size

        self.params = [param for layer in self.layers for param in layer.params]

        self.x = tensor.matrix("x")
        self.y = tensor.ivector("y")

        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)

        for i in range(1, len(self.layers)):
            prev_layer, layer = self.layers[i - 1], self.layers[i]
            layer.set_inpt(prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)

        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def stochastic_gradient_descent(
        self,
        training_data,
        epochs,
        eta,
        validation_data,
        test_data,
        lmbda=0.0,
    ):
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        training_mini_batch_count = data_size(training_data) // self.mini_batch_size
        validation_mini_batch_count = data_size(validation_data) // self.mini_batch_size
        test_mini_batch_count = data_size(test_data) // self.mini_batch_size

        l2_regularization = sum([(layer.w**2).sum() for layer in self.layers])

        cost = self.layers[-1].cost(self) + 0.5 * lmbda * l2_regularization / training_mini_batch_count
        gradients = tensor.grad(cost, self.params)

        pytensor.pp(cost)

        updates = [(param, param - eta * gradient) for param, gradient in zip(self.params, gradients)]

        mini_batch_index = tensor.lscalar()

        train_mini_batch = pytensor.function(
            [mini_batch_index],
            cost,
            updates=updates,
            givens={
                self.x: training_x[
                    mini_batch_index * self.mini_batch_size : (mini_batch_index + 1) * self.mini_batch_size
                ],
                self.y: training_y[
                    mini_batch_index * self.mini_batch_size : (mini_batch_index + 1) * self.mini_batch_size
                ],
            },
            # mode=pytensor.compile.MonitorMode(pre_func=inspect_inputs, post_func=inspect_outputs),
        )

        validate_mini_batch = pytensor.function(
            [mini_batch_index],
            self.layers[-1].accuracy(self.y),
            givens={
                self.x: validation_x[
                    mini_batch_index * self.mini_batch_size : (mini_batch_index + 1) * self.mini_batch_size
                ],
                self.y: validation_y[
                    mini_batch_index * self.mini_batch_size : (mini_batch_index + 1) * self.mini_batch_size
                ],
            },
        )

        test_mini_batch = pytensor.function(
            [mini_batch_index],
            self.layers[-1].accuracy(self.y),
            givens={
                self.x: test_x[mini_batch_index * self.mini_batch_size : (mini_batch_index + 1) * self.mini_batch_size],
                self.y: test_y[mini_batch_index * self.mini_batch_size : (mini_batch_index + 1) * self.mini_batch_size],
            },
        )

        self.test_mini_batch_predictions = pytensor.function(
            [mini_batch_index],
            self.layers[-1].y_out,
            givens={
                self.x: test_x[mini_batch_index * self.mini_batch_size : (mini_batch_index + 1) * self.mini_batch_size]
            },
        )

        best_iteration = 0
        test_accuracy = 0

        best_validation_accuracy = 0.0
        for epoch in range(epochs):
            for mb_index in range(training_mini_batch_count):
                iteration = training_mini_batch_count * epoch + mb_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost = train_mini_batch(mb_index)
                if (iteration + 1) % training_mini_batch_count == 0:
                    validation_accuracy = np.mean([validate_mini_batch(j) for j in range(validation_mini_batch_count)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean([test_mini_batch(j) for j in range(test_mini_batch_count)])
                            print("The corresponding test accuracy is {0:.2%}".format(test_accuracy))

        print("Finished training network.")
        print(
            "Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
                best_validation_accuracy, best_iteration
            )
        )
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))


def data_size(data):
    return data[0].get_value(borrow=True).shape[0]


def dropout_layer(layer, p_dropout):
    srng = RandomStream(np.random.default_rng(seed=0).integers(999999))
    mask = srng.binomial(n=1, p=1 - p_dropout, size=layer.shape)
    return layer * tensor.cast(mask, pytensor.config.floatX)


def inspect_inputs(fgraph, i, node, fn):
    print(i, node, "input(s) value(s):", [inpt[0] for inpt in fn.inputs], end="")


def inspect_outputs(fgraph, i, node, fn):
    print(" output(s) value(s):", [output[0] for output in fn.outputs])
