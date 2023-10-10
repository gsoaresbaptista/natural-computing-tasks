import matplotlib.pyplot as plt
import numpy as np
from natural_computing import (
    LayerFactory,
    MinMaxScaler,
    NeuralNetwork,
    create_window,
    rmse,
    split_train_test,
    fetch_file_and_convert_to_array,
)

# experiment settings
plot_result = False
curve_plot = 'test_curve'  # train_curve or test_curve
window_size = 7
batch_size = 32
n_epochs = 10
train_percentage = 0.8
data_path = (
    'https://raw.githubusercontent.com/gsoaresbaptista/'
    'natural-computing/main/data/prediction'
)


if __name__ == '__main__':
    # read only temperatures
    data = fetch_file_and_convert_to_array(
        f'{data_path}/daily-max-temperatures.csv', skiprows=1, usecols=[1]
    ).squeeze()

    # create window to make prediction
    x, y = create_window(data, window_size=window_size)
    (x_train, y_train), (x_test, y_test) = split_train_test(
        x, y, train_percentage, sequential=True
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
                64,
                1,
                activation='linear',
                weights_initializer='glorot_normal',
                biases_initializer='glorot_normal',
                regularization_strength=1e-4,
            ),
        ]
    )

    # fit network
    model.fit(
        x_train_shuffled,
        y_train_shuffled,
        epochs=n_epochs,
        verbose=100,
        batch_size=batch_size,
    )
    model.save('model_prediction.pkl')

    if plot_result:
        # plot train or test curve
        plt.plot(range(len(y_train)), y_train, c='green')

        if curve_plot == 'test_curve':
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

        if curve_plot == 'train_curve':
            plt.plot(
                range(len(y_train)),
                y_train,
                c='blue',
                label='real',
            )
            plt.plot(
                range(len(y_train)),
                model.predict(x_train_std),
                c='red',
                label='predicted',
            )

        plt.legend()
        plt.show()
