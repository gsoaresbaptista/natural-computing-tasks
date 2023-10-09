from typing import Tuple, List, Dict, Any
import numpy as np


from natural_computing import NeuralNetwork, LayerFactory


def create_window(
    data: np.ndarray, window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    x_data, y_data = [], []
    dataset_size = len(data)

    for i in range(dataset_size):
        # check if there is enough data
        if i + window_size + 1 > dataset_size:
            break

        # append data
        x_data.append(data[i: i + window_size])
        y_data.append(data[i + window_size])

    return (np.array(x_data), np.array(y_data).reshape(-1, 1))


def split_train_test(
    x: np.ndarray, y: np.ndarray, train_size: float, sequential: bool = False
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    # calculate the size of the sets
    data_size = len(x)
    train_size = int(train_size * data_size)

    # generate indices
    if not sequential:
        train_indices = np.random.choice(
            range(data_size), size=train_size, replace=False
        )
        test_indices = np.array(
            [i for i in range(data_size) if i not in train_indices]
        )
    else:
        train_indices = np.array(range(train_size))
        test_indices = np.array(range(train_size, data_size))

    return (
        (x[train_indices], y[train_indices]),
        (x[test_indices], y[test_indices]),
    )


class StandardScaler:
    def __init__(self, centered_on_zero: bool = True) -> None:
        self._mean = 0
        self._std = 0
        self._fitted = False
        self._centered_on_zero = centered_on_zero

    def fit(self, x: np.ndarray) -> None:
        self._mean = np.mean(x)
        self._std = np.std(x)
        self._fitted = True

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_transformed = (x - self._mean) / self._std

        if not self._centered_on_zero:
            x_transformed = 2 * x_transformed - 1

        return x_transformed

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


class MinMaxScaler:
    def __init__(self, centered_on_zero: bool = True) -> None:
        self._min = 0
        self._max = 0
        self._fitted = False
        self._centered_on_zero = centered_on_zero

    def fit(self, x: np.ndarray) -> None:
        self._min = np.min(x)
        self._max = np.max(x)
        self._fitted = True

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_transformed = (x - self._min) / (self._max - self._min)

        if not self._centered_on_zero:
            x_transformed = 2 * x_transformed - 1

        return x_transformed

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


def unpack_network(
    model: NeuralNetwork,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    weights, decode_guide = [], []

    # unpack all layers
    for layer in model._layers:
        weights.append(layer._weights.reshape(-1, 1))
        weights.append(layer._biases.reshape(-1, 1))
        decode_guide.append(
            {
                'weights_shape': layer._weights.shape,
                'biases_shape': layer._biases.shape,
                'activation': layer._activation,
            }
        )

    return np.concatenate(weights), decode_guide


def pack_network(
    weights_vector: np.ndarray, decode_guide: List[Dict[str, Any]]
) -> NeuralNetwork:
    module = NeuralNetwork(0)

    for layer in decode_guide:
        x, y = layer['weights_shape']
        a, b = layer['biases_shape']

        # unpack weights
        weights = weights_vector[: x * y].reshape((x, y))
        weights_vector = weights_vector[x * y:]

        # unpack biases
        biases = weights_vector[: a * b].reshape((a, b))
        weights_vector = weights_vector[a * b:]

        # add layer to network
        activation = layer['activation'].__name__
        module.add_layer(
            LayerFactory.dense_layer(
                weights.shape[1], weights.shape[0], activation=activation
            )
        )

        # change weights
        module._layers[-1]._weights = weights
        module._layers[-1]._biases = biases

    return module
