import numpy as np
from evolutionary_programming.data_processing import (
    MinMaxScaler,
    fetch_csv_to_numpy,
    split_train_test,
)
from evolutionary_programming.neural_network import (
    DenseLayer,
    NeuralNetwork,
    encode_neural_network,
)
from sklearn.preprocessing import OneHotEncoder

iris_url = (
    'https://raw.githubusercontent.com/gsoaresbaptista/'
    'datasets/main/datasets/iris.csv'
)

np.random.seed(42)

PREDICTION_REGULARIZATION = 0.0
REGRESSION_REGULARIZATION = 1e-4


class NeuralNetworkArchitectures:
    @staticmethod
    def iris_architecture() -> NeuralNetwork:
        module = NeuralNetwork(
            1e-3, loss_function='softmax_neg_log_likelihood', momentum=0.9
        )
        module.add_layer(
            [
                DenseLayer(
                    4,
                    10,
                    'tanh',
                    regularization_strength=PREDICTION_REGULARIZATION,
                ),
                DenseLayer(
                    10,
                    3,
                    'linear',
                    regularization_strength=PREDICTION_REGULARIZATION,
                ),
            ]
        )
        return module


class DecodeGuides:
    @staticmethod
    def iris_guide() -> list:
        _, guide = encode_neural_network(
            NeuralNetworkArchitectures.prediction_architecture()
        )
        return guide


class DatasetsDownloader:
    @staticmethod
    def iris() -> np.ndarray:
        # download data
        data = fetch_csv_to_numpy(iris_url, header=False)

        # split data into x and y
        one_hot = OneHotEncoder(sparse_output=False)
        x = np.concatenate(data[:-1], axis=1)
        y = one_hot.fit_transform(data[-1].reshape(-1, 1))

        #
        ((x_train, y_train), (x_test, y_test)) = split_train_test(
            x, y, train_percentage=0.8, sequential=False
        )

        # preprocess data
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        size, rnd = len(x_train), np.random.RandomState(0)
        indices = rnd.choice(range(size), size, replace=False)
        x_train = x_train[indices]
        y_train = y_train[indices]

        return {
            'indices': np.argsort(indices),
            'processed': ((x_train, y_train), (x_test, y_test)),
            'inverse_fn': lambda x: scaler.inverse_transform(x),
        }
