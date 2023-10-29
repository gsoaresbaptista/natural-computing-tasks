import numpy as np
from evolutionary_programming.data_processing import (
    MinMaxScaler,
    create_window,
    fetch_csv_to_numpy,
    split_train_test,
)
from evolutionary_programming.neural_network import (
    DenseLayer,
    NeuralNetwork,
    encode_neural_network,
)

temperatures_url = (
    'https://raw.githubusercontent.com/gsoaresbaptista/'
    'datasets/main/datasets/temperatures.csv'
)


regression_train_url = (
    'https://raw.githubusercontent.com/gsoaresbaptista/'
    'datasets/main/datasets/outlier_regression/train.csv'
)

regression_test_url = (
    'https://raw.githubusercontent.com/gsoaresbaptista/'
    'datasets/main/datasets/outlier_regression/test.csv'
)


WINDOW_SIZE = 3
PREDICTION_REGULARIZATION = 0.0
REGRESSION_REGULARIZATION = 1e-4


class NeuralNetworkArchitectures:
    @staticmethod
    def prediction_architecture() -> NeuralNetwork:
        module = NeuralNetwork(1e-3, loss_function='rmse', momentum=0.9)
        module.add_layer(
            [
                DenseLayer(
                    WINDOW_SIZE, 10, 'sigmoid',
                    regularization_strength=PREDICTION_REGULARIZATION),
                DenseLayer(
                    10, 1, 'linear',
                    regularization_strength=PREDICTION_REGULARIZATION),
            ]
        )
        return module

    @staticmethod
    def regression_architecture() -> NeuralNetwork:
        module = NeuralNetwork(1e-3, loss_function='rmse', momentum=0.9)
        module.add_layer(
            [
                DenseLayer(
                    1, 10, 'tanh',
                    regularization_strength=REGRESSION_REGULARIZATION),
                DenseLayer(
                    10, 1, 'linear',
                    regularization_strength=REGRESSION_REGULARIZATION),
            ]
        )
        return module


class DecodeGuides:
    @staticmethod
    def prediction_guide() -> list:
        _, guide = encode_neural_network(
            NeuralNetworkArchitectures.prediction_architecture()
        )
        return guide

    @staticmethod
    def regression_guide() -> list:
        _, guide = encode_neural_network(
            NeuralNetworkArchitectures.regression_architecture()
        )
        return guide


class DatasetsDownloader:
    @staticmethod
    def prediction(temperature: str) -> np.ndarray:
        # download data
        column_names, temperatures = fetch_csv_to_numpy(
            temperatures_url, columns=[1 if temperature == 'min' else 2]
        )

        # preprocess data
        scaler = MinMaxScaler()
        scaler.fit(temperatures)
        temperatures = scaler.transform(temperatures)

        x, y = create_window(temperatures, WINDOW_SIZE)
        (x_train, y_train), (x_test, y_test) = split_train_test(
            x, y, train_percentage=0.8, sequential=True
        )

        size, rnd = len(x_train), np.random.RandomState(0)
        indices = rnd.choice(range(size), size, replace=False)
        x_train = x_train[indices]
        y_train = y_train[indices]

        return {
            'indices': np.argsort(indices),
            'processed': ((x_train, y_train), (x_test, y_test)),
            'inverse_fn': lambda x: scaler.inverse_transform(x),
        }

    @staticmethod
    def regression() -> np.ndarray:
        # download data
        column_names, x_train, y_train = fetch_csv_to_numpy(
            regression_train_url, columns=[0, 1]
        )
        column_names, x_test = fetch_csv_to_numpy(
            regression_test_url, columns=[0]
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
            'processed': ((x_train, y_train), (x_test, None)),
            'inverse_fn': lambda x: scaler.inverse_transform(x),
        }
