from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from natural_computing import BaseFunction, LayerFactory, NeuralNetwork
from natural_computing.optimization import (
    BareBonesParticleSwarmOptimization,
)

from utils import (
    MinMaxScaler,
    pack_network,
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
        error = np.sqrt(
            np.mean((nn.predict(self.x_data) - self.y_data) ** 2)
        ) + 1e-4 * np.sum(
            np.concatenate([layer._weights.squeeze() for layer in nn._layers])
            ** 2
        )
        return error


if __name__ == '__main__':
    nn = NeuralNetwork(0)
    nn.add_layer(
        [
            LayerFactory.dense_layer(
                1,
                10,
                activation='sigmoid',
            ),
            LayerFactory.dense_layer(
                10,
                1,
                activation='linear',
            ),
        ]
    )

    # get decode guide
    _, decode_guide = unpack_network(nn)

    # data
    x_train = np.loadtxt('data/regression/x_treinamento.txt').reshape((-1, 1))
    y_train = np.loadtxt('data/regression/y_treinamento.txt').reshape((-1, 1))
    x_test = np.loadtxt('data/regression/x_teste.txt').reshape((-1, 1))

    # scaler
    scaler = MinMaxScaler(centered_on_zero=False)
    scaler.fit(x_train)
    x_train_std = scaler.transform(x_train)
    x_test_std = scaler.transform(x_test)

    # shuffle data
    indices = np.random.choice(
        range(x_train.shape[0]), x_train.shape[0], replace=False
    )
    x_train_shuffled = x_train_std[indices]
    y_train_shuffled = y_train[indices]

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
    pso = BareBonesParticleSwarmOptimization(
        20, 5000, [[-1.0, 1.0] for _ in range(individual_shape)]
    )

    pso.optimize(rmse)

    # get best model
    best_weights = np.array(pso.best_global_position).reshape(-1, 1)
    model = pack_network(best_weights, decode_guide)

    # plot curves
    plt.scatter(x_train, y_train, c='blue')
    plt.scatter(x_test, model.predict(x_test_std), c='green')
    plt.plot(x_test, model.predict(x_test_std), c='red')
    plt.show()
