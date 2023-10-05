import matplotlib.pyplot as plt
import numpy as np
from natural_computing.neural_network import (
    NeuralNetwork,
    rmse,
)
from natural_computing.utils import LayerFactory

from utils import MinMaxScaler, create_window, split_train_test

np.random.seed(42)


if __name__ == '__main__':
    # read only temperatures
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
        range(len(x_train)), len(x_train), replace=False
    )
    x_train_shuffled = x_train_std[shuffle_indices]
    y_train_shuffled = y_train[shuffle_indices]

    # create neural network
    model = NeuralNetwork(
        learning_rate=1e-2,
        loss_function=rmse,
        momentum=0.99,
    )
    model.add_layer(
        [
            LayerFactory.dense_layer(
                window_size,
                64,
                activation='tanh',
                weights_initializer='glorot_normal',
                biases_initializer='glorot_normal',
                regularization_strength=1e-4,
            ),
            LayerFactory.dense_layer(
                64, 1, activation='linear', regularization_strength=1e-4
            ),
        ]
    )

    # fit network
    model.fit(
        x_train_shuffled,
        y_train_shuffled,
        epochs=3000,
        verbose=100,
        batch_size=32,
    )
    model.save('model_prediction.pkl')

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
