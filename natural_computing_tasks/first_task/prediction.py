from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from natural_computing import BaseFunction, LayerFactory, NeuralNetwork
from natural_computing.optimization import (
    ParticleSwarmOptimization,
)

from utils import (
    MinMaxScaler,
    create_window,
    pack_network,
    split_train_test,
    unpack_network,
)


class RootMeanSquaredError(BaseFunction):
    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        decode_guide: List[Tuple[int, int]],
    ):
        super().__init__()
        self.x_data = x_data
        self.y_data = y_data
        self.decode_guide = decode_guide

    def evaluate(self, nn_weights: List[float]) -> float:
        nn_weights = np.array(nn_weights).reshape(-1, 1)
        nn = pack_network(nn_weights, self.decode_guide)
        error: float = np.sqrt(
            np.mean((self.y_data - nn.predict(self.x_data)) ** 2)
        )
        return error


if __name__ == '__main__':
    nn = NeuralNetwork(0)
    nn.add_layer(
        [
            LayerFactory.dense_layer(7, 64, activation='tanh'),
            LayerFactory.dense_layer(64, 1, activation='linear'),
        ]
    )

    # get decode guide
    vector, decode_guide = unpack_network(nn)

    # data
    file_path = 'data/prediction/daily-max-temperatures.csv'
    data = np.loadtxt(
        file_path, delimiter=',', skiprows=1, usecols=(1,), dtype=float
    )

    # create window to make prediction
    window_size = 7

    x, y = create_window(data, window_size=window_size)
    (x_train, y_train), (x_test, y_test) = split_train_test(
        x, y, 0.8, sequential=True
    )

    # min max scaler in range [-1, 1]
    scaler = MinMaxScaler(centered_on_zero=False)
    scaler.fit(x_train)

    # scale data
    x_train_std = scaler.transform(x_train)
    x_test_std = scaler.transform(x_test)

    # shuffle data
    shuffle_indices = np.random.choice(
        range(x_train.shape[0]), x_train.shape[0], replace=False
    )
    x_train_shuffled = x_train_std[shuffle_indices]
    y_train_shuffled = y_train[shuffle_indices]

    # create pso structure
    individual_shape = sum(
        [
            layer_dict['weights_shape'][0] * layer_dict['weights_shape'][1]
            + layer_dict['biases_shape'][0] * layer_dict['biases_shape'][1]
            for layer_dict in decode_guide
        ]
    )
    rmse = RootMeanSquaredError(
        x_train_shuffled, y_train_shuffled, decode_guide
    )
    pso = ParticleSwarmOptimization(
        20, 1000, 0.8, 0.5, 0.5, [[-1, 1] for _ in range(individual_shape)]
    )
    pso.optimize(rmse)

    # get best model
    best_weights = np.array(pso.best_global_position).reshape(-1, 1)
    model = pack_network(best_weights, decode_guide)

    # plot curves
    plt.plot(range(len(y_train)), y_train, c='green')
    plt.plot(
        range(len(y_train), len(y_train) + len(y_test)),
        y_test,
        c='blue',
        label='real',
    )
    plt.plot(
        range(len(y_train), len(y_train) + len(y_test)),
        model.predict(x_test_std),
        c='red',
        label='predicted',
    )
    plt.legend()
    plt.show()
