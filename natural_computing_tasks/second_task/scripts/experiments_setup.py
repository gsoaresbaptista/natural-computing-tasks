import numpy as np
import pandas as pd
from evolutionary_programming.data_processing import (
    fetch_csv_to_numpy,
    split_train_test,
)
from evolutionary_programming.neural_network import (
    DenseLayer,
    NeuralNetwork,
    encode_neural_network,
)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler

iris_url = (
    'https://raw.githubusercontent.com/gsoaresbaptista/'
    'datasets/main/datasets/iris.csv'
)

hepatitis_url = (
    'https://raw.githubusercontent.com/gsoaresbaptista/'
    'datasets/main/datasets/hepatitis.csv'
)

heart_url = (
    'https://raw.githubusercontent.com/gsoaresbaptista/'
    'datasets/main/datasets/heart_disease.csv'
)


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
                    5,
                    'tanh',
                    regularization_strength=PREDICTION_REGULARIZATION,
                ),
                DenseLayer(
                    5,
                    3,
                    'linear',
                    regularization_strength=PREDICTION_REGULARIZATION,
                ),
            ]
        )
        return module

    @staticmethod
    def hepatitis_architecture() -> NeuralNetwork:
        module = NeuralNetwork(
            1e-3, loss_function='softmax_neg_log_likelihood', momentum=0.9
        )
        module.add_layer(
            [
                DenseLayer(
                    16,
                    20,
                    'tanh',
                    regularization_strength=PREDICTION_REGULARIZATION,
                ),
                DenseLayer(
                    20,
                    2,
                    'linear',
                    regularization_strength=PREDICTION_REGULARIZATION,
                ),
            ]
        )
        return module

    @staticmethod
    def heart_architecture() -> NeuralNetwork:
        module = NeuralNetwork(
            1e-3, loss_function='softmax_neg_log_likelihood', momentum=0.9
        )
        module.add_layer(
            [
                DenseLayer(
                    13,
                    20,
                    'tanh',
                    regularization_strength=PREDICTION_REGULARIZATION,
                ),
                DenseLayer(
                    20,
                    2,
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
            NeuralNetworkArchitectures.iris_architecture()
        )
        return guide

    @staticmethod
    def hepatitis_guide() -> list:
        _, guide = encode_neural_network(
            NeuralNetworkArchitectures.hepatitis_architecture()
        )
        return guide

    @staticmethod
    def heart_guide() -> list:
        _, guide = encode_neural_network(
            NeuralNetworkArchitectures.heart_architecture()
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

    @staticmethod
    def hepatitis() -> np.ndarray:
        # download data
        csv = fetch_csv_to_numpy(hepatitis_url, header=True, convert=False)
        data = np.array(csv[1:]).squeeze()
        columns = csv[0]
        df = pd.DataFrame(data.T, columns=columns)

        # imputing data
        df = df.replace('', np.nan)
        df = df.drop(['protime', 'alk_phosphate', 'albumin'], axis=1)
        df = df.dropna()
        df = df.astype('string')
        df = df.astype({'age': 'float', 'bilirubin': 'float', 'sgot': float})

        # encoder values
        columns_to_encoder = df.columns[(df.map(type) == str).all(0)]

        for column in columns_to_encoder:
            df[column] = LabelEncoder().fit_transform(df[column])

        df = df.reset_index()

        x = df.iloc[:, 1:-1].values
        y = df.iloc[:, -1].values

        # split data
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
        y_train = y_train[indices].reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        return {
            'indices': np.argsort(indices),
            'processed': ((x_train, y_train), (x_test, y_test)),
            'inverse_fn': lambda x: scaler.inverse_transform(x),
        }

    @staticmethod
    def heart() -> np.ndarray:
        # download data
        csv = fetch_csv_to_numpy(heart_url, header=True, convert=True)
        data = np.array(csv[1:]).squeeze()
        columns = csv[0]
        df = pd.DataFrame(data.T, columns=columns)

        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # split data
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
        y_train = y_train[indices].reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        return {
            'indices': np.argsort(indices),
            'processed': ((x_train, y_train), (x_test, y_test)),
            'inverse_fn': lambda x: scaler.inverse_transform(x),
        }
