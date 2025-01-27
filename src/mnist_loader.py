import gzip
import os.path
import pickle
from typing import Any

import numpy as np
import pytensor
from PIL import Image


def load_raw_data_to_array(image_file, label_file):
    extracted_label_file = gzip.open(label_file, "rb")
    extracted_image_file = gzip.open(image_file, "rb")

    # Skip the magic number
    extracted_label_file.read(4)
    extracted_image_file.read(4)

    number_of_labels = int.from_bytes(extracted_label_file.read(4), byteorder="big")
    number_of_images = int.from_bytes(extracted_image_file.read(4), byteorder="big")

    if number_of_images != number_of_labels:
        raise Exception("Number of images not the same as the number of labels")

    image_rows = int.from_bytes(extracted_image_file.read(4), byteorder="big")
    image_columns = int.from_bytes(extracted_image_file.read(4), byteorder="big")

    labels = np.frombuffer(extracted_label_file.read(number_of_labels), dtype=np.uint8)
    images = np.frombuffer(
        extracted_image_file.read(number_of_images * image_rows * image_columns),
        dtype=np.uint8,
    )
    images = images.reshape(number_of_images, image_rows, image_columns)

    extracted_image_file.close()
    extracted_label_file.close()

    return labels, images, image_rows, image_columns


def pickle_data(
    labels_training,
    images_training,
    labels_validation,
    images_validation,
    labels_test,
    images_test,
    pickle_file,
    image_rows,
    image_columns,
):
    training_data = (
        np.reshape(
            images_training / 255,
            (len(images_training), image_rows * image_columns),
        ),
        labels_training,
    )
    validation_data = (
        np.reshape(
            images_validation / 255,
            (len(images_validation), image_rows * image_columns),
        ),
        labels_validation,
    )
    test_data = (
        np.reshape(
            images_test / 255,
            (len(images_test), image_rows * image_columns),
        ),
        labels_test,
    )

    with gzip.open(pickle_file, "wb") as file:
        # noinspection PyTypeChecker
        pickle.dump((training_data, validation_data, test_data), file)


def save_images(image_tuple, image_root):
    for i, (label, image_matrix) in enumerate(image_tuple):
        image = Image.fromarray(image_matrix)

        path = os.path.join(image_root, str(i) + "-" + str(label) + ".png")
        image.save(path)


def load_pickle_data(pickle_file):
    file = gzip.open(pickle_file, "rb")
    training_data, validation_data, test_data = pickle.load(file)
    file.close()
    return training_data, validation_data, test_data


def load_pickle_data_wrapper(pickle_file):
    training_data, validation_data, test_data = load_pickle_data(pickle_file)

    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_results = [vectorised_result(y) for y in training_data[1]]
    training_data_result = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_data_result = zip(validation_inputs, validation_data[1])

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data_result = zip(test_inputs, test_data[1])

    return training_data_result, validation_data_result, test_data_result


def load_pickle_data_shared(pickle_file):
    training_data, validation_data, test_data = load_pickle_data(pickle_file)

    def shared(data):
        shared_x = pytensor.shared(np.asarray(data[0], dtype=pytensor.config.floatX), borrow=True)
        shared_y = pytensor.shared(np.asarray(data[1], dtype=pytensor.config.floatX), borrow=True)
        return shared_x, pytensor.tensor.cast(shared_y, "int32")

    return shared(training_data), shared(validation_data), shared(test_data)


def vectorised_result(j):
    result = np.zeros((10, 1))
    result[j] = 1.0
    return result


def expand(image_tuple, image_rows, image_columns):
    expanded_pairs: list[Any] = []
    i = 0
    for label, image in image_tuple:
        expanded_pairs.append((label, image))
        if (i + 1) % 1000 == 0:
            print("Expanding image number {0}", i + 1)

        for displacement, axis, index in [
            (1, 0, 0),  # Vertically 1 step to the right
            (-1, 0, image_columns),  # Vertically 1 step to the left
            (1, 1, 0),  # Horizontally 1 step to the bottom
            (-1, 1, image_rows),  # Horizontally 1 step to the top
        ]:
            # rolls the matrix in a direction bringing the edge elements to the other side
            displaced_image = np.roll(image, displacement, axis)

            # Ensures that the rolled over row or column doesn't contain any pixels
            if axis == 0:
                if displacement > 0:
                    displaced_image[index : index + displacement, :] = np.zeros((displacement, image_columns))
                else:
                    displaced_image[index + displacement : index, :] = np.zeros((-displacement, image_columns))
            else:
                if displacement > 0:
                    displaced_image[:, index : index + displacement] = np.zeros((image_rows, displacement))
                else:
                    displaced_image[:, index + displacement : index] = np.zeros((image_rows, -displacement))

            expanded_pairs.append((label, displaced_image))

        i += 1

    np.random.default_rng().shuffle(expanded_pairs)

    labels, images = zip(*expanded_pairs)

    return np.array(labels), np.array(images)
