import matplotlib.pyplot as plt
import numpy as np
from natural_computing import (
    LayerFactory,
    NeuralNetwork,
    fetch_file_and_convert_to_array,
    learning_rate_staircase_decay,
    rmse,
    MinMaxScaler,
)

# experiment settings
plot_result = False
curve_plot = 'test_curve'  # train_curve or test_curve
data_path = (
    'https://raw.githubusercontent.com/gsoaresbaptista/'
    'natural-computing/main/data/regression'
)


if __name__ == '__main__':
    x_train = fetch_file_and_convert_to_array(f'{data_path}/x_train.txt')
    y_train = fetch_file_and_convert_to_array(f'{data_path}/y_train.txt')
    input_dim, output_dim = 1, 1

    # min max scaling
    x_train_std = MinMaxScaler(centered_on_zero=False).fit_transform(x_train)

    # shuffle data
    indices = np.random.randint(0, x_train.shape[0], x_train.shape[0])
    x_shuffled, y_shuffled = x_train_std[indices], y_train[indices]

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

    x_test = fetch_file_and_convert_to_array(f'{data_path}/x_test.txt')

    x_test_std = (x_test - x_train.min(axis=0)) / (
        x_train.max(axis=0) - x_train.min(axis=0)
    )
    x_test_std = 2 * x_test_std - 1

    if plot_result:
        # plot train or test curve
        plt.scatter(x_train, y_train)

        if curve_plot == 'train_curve':
            plt.scatter(x_train, nn.predict(x_train_std), c='red')

        if curve_plot == 'test_curve':
            plt.scatter(x_test, nn.predict(x_test_std), c='red')
            plt.plot(x_test, nn.predict(x_test_std), c='green')

        plt.show()  # type: ignore
