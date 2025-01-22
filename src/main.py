import mnist_loader
import network
import network2


def main():
    # Uncomment to load the raw binary data to numpy arrays
    # labels_training, images_training, image_rows, image_columns = mnist_loader.load_raw_data_to_array(
    #     "./data/train-images-idx3-ubyte.gz", "./data/train-labels-idx1-ubyte.gz"
    # )
    #
    # labels_test, images_test, image_rows, image_columns = mnist_loader.load_raw_data_to_array(
    #     "./data/t10k-images-idx3-ubyte.gz", "./data/t10k-labels-idx1-ubyte.gz"
    # )

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

    # Loads the data from the pickle file in the useful structure
    training_data, validation_data, test_data = mnist_loader.load_pickle_data_wrapper("./data/mnist.pkl.gz")

    # net = network.Network([784, 30, 10])
    # Run the stochastic gradient descent algorithm and test the training over test data
    # net.stochastic_gradient_descent(list(training_data), 30, 10, 3.0, list(test_data))

    net2 = network2.Network([784, 30, 10])
    net2.stochastic_gradient_descent(
        list(training_data), 30, 10, 0.5, 5.0, list(validation_data),
        True,
        True,
        True,
        True
    )


main()
