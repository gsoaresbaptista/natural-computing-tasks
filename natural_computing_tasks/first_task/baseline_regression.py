import matplotlib.pyplot as plt
import numpy as np

from natural_computing.utils import LayerFactory
from natural_computing.neural_network import (
    NeuralNetwork,
    rmse,
    learning_rate_staircase_decay,
)

np.random.seed(42)


if __name__ == '__main__':
    x_train = np.loadtxt('data/regression/x_treinamento.txt').reshape((-1, 1))
    y_train = np.loadtxt('data/regression/y_treinamento.txt').reshape((-1, 1))
    input_dim, output_dim = 1, 1

    # min max scaling
    x_std = (x_train - x_train.min(axis=0)) / (
        x_train.max(axis=0) - x_train.min(axis=0)
    )
    x_std = 2 * x_std - 1

    # shuffle data
    indices = np.random.randint(0, x_train.shape[0], x_train.shape[0])
    x_shuffled, y_shuffled = x_std[indices], y_train[indices]

    nn = NeuralNetwork(
        learning_rate=1e-1,
        loss_function=rmse,
        lr_decay_rate=0.1,
        lr_decay_fn=learning_rate_staircase_decay,
        lr_decay_steps=500,
        momentum=0.9,
    )
    nn.add_layer(
        [
            LayerFactory.dense_layer(input_dim, 10, 'sigmoid'),
            LayerFactory.dense_layer(10, output_dim, 'linear'),
        ]
    )

    nn.fit(
        x_shuffled,
        y_shuffled,
        epochs=1000,
        batch_size=x_train.shape[0],
        verbose=50,
    )

    nn.save('best_model.pkl')

    x_test = np.loadtxt('data/regression/x_teste.txt').reshape((-1, 1))
    x_test_std = (x_test - x_train.min(axis=0)) / (
        x_train.max(axis=0) - x_train.min(axis=0)
    )
    x_test_std = 2 * x_test_std - 1

    plt.scatter(x_train, y_train)
    # plt.scatter(x_test, nn.predict(x_test_std), c='red')
    plt.scatter(x_train, nn.predict(x_std), c='red')
    # plt.plot(x_test, nn.predict(x_test_std), c='green')
    plt.show()
