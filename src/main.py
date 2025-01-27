import mnist_loader
import network3


def main():
    # Uncomment to load the raw binary data to numpy arrays
    # labels_training, images_training, image_rows, image_columns = mnist_loader.load_raw_data_to_array(
    #     "./data/train-images-idx3-ubyte.gz", "./data/train-labels-idx1-ubyte.gz"
    # )
    #
    # labels_test, images_test, image_rows, image_columns = mnist_loader.load_raw_data_to_array(
    #     "./data/t10k-images-idx3-ubyte.gz", "./data/t10k-labels-idx1-ubyte.gz"
    # )

    # labels_training_expanded, images_training_expanded = mnist_loader.expand(zip(labels_training[:50000], images_training[:50000]), image_rows, image_columns)

    # Uncomment if the images needs to be created
    # mnist_loader.save_images(zip(labels_training, images_training), "./data/images/train")
    # mnist_loader.save_images(zip(labels_test, images_test), "./data/images/test")

    # Uncomment to the generate the pickle file from the raw binary data
    # mnist_loader.pickle_data(
    #     labels_training[:50000],
    #     images_training[:50000],
    #     labels_training[-10000:],
    #     images_training[-10000:],
    #     labels_test,
    #     images_test,
    #     "./data/mnist.pkl.gz",
    #     image_rows,
    #     image_columns,
    # )

    # mnist_loader.pickle_data(
    #     labels_training_expanded,
    #     images_training_expanded,
    #     labels_training[-10000:],
    #     images_training[-10000:],
    #     labels_test,
    #     images_test,
    #     "./data/mnist_expanded.pkl.gz",
    #     image_rows,
    #     image_columns,
    # )

    # Loads the data from the pickle file in the useful structure
    training_data, validation_data, test_data = mnist_loader.load_pickle_data_wrapper("./data/mnist.pkl.gz")
    training_data_shared, validation_data_shared, test_data_shared = mnist_loader.load_pickle_data_shared(
        "./data/mnist.pkl.gz"
    )
    # training_data_shared, _, _ = mnist_loader.load_pickle_data_shared("./data/mnist_expanded.pkl.gz")

    # net = network.Network([784, 30, 10])
    # Run the stochastic gradient descent algorithm and test the training over test data
    # net.stochastic_gradient_descent(list(training_data), 30, 10, 3.0, list(test_data))

    # net2 = network2.Network([784, 30, 10])
    # net2.stochastic_gradient_descent(
    #     training_data=list(training_data),
    #     epochs=30,
    #     mini_batch_size=10,
    #     initial_eta=0.5,
    #     lmbda=5.0,
    #     momentum_coefficient=0.1,
    #     early_stopping=10,
    #     learning_rate_schedule=128,
    #     evaluation_data=list(validation_data),
    #     monitor_evaluation_cost=True,
    #     monitor_evaluation_accuracy=True,
    #     monitor_training_cost=True,
    #     monitor_training_accuracy=True,
    # )

    net3 = network3.Network(
        [network3.FullyConnectedLayer(n_in=784, n_out=100), network3.SoftmaxLayer(n_in=100, n_out=10)],
        mini_batch_size=10,
        training_data=list(training_data)[:10],
    )

    net3.stochastic_gradient_descent(
        training_data=training_data_shared,
        epochs=60,
        eta=0.1,
        validation_data=validation_data_shared,
        test_data=test_data_shared,
    )


main()
