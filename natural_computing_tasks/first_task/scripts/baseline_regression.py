import argparse
import matplotlib.pyplot as plt
import numpy as np
from natural_computing import (
    LayerFactory,
    NeuralNetwork,
    fetch_file_and_convert_to_array,
    learning_rate_staircase_decay,
    rmse,
    MinMaxScaler,
    r2_score,
)

# experiment settings
plot_result = False
curve_plot = 'test_curve'  # train_curve or test_curve
batch_size = 320
n_epochs = 5000
data_path = (
    'https://raw.githubusercontent.com/gsoaresbaptista/'
    'natural-computing/main/data/regression'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Neural Network Training Script'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Filepath to save the trained model',
    )
    parser.add_argument(
        '--return-best-loss',
        action='store_true',
        help='Return the best loss from algorithm',
    )
    parser.add_argument(
        '--return-r2-score',
        action='store_true',
        help='Return the r2 score',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    x_train = fetch_file_and_convert_to_array(f'{data_path}/x_train.txt')
    y_train = fetch_file_and_convert_to_array(f'{data_path}/y_train.txt')
    x_test = fetch_file_and_convert_to_array(f'{data_path}/x_test.txt')

    input_dim, output_dim = 1, 1

    # min max scaling
    scaler = MinMaxScaler(centered_on_zero=False)
    scaler.fit(x_train)

    x_train_std = scaler.transform(x_train)
    x_test_std = scaler.transform(x_test)

    # shuffle data
    indices = np.random.randint(0, x_train.shape[0], x_train.shape[0])
    x_shuffled, y_shuffled = x_train_std[indices], y_train[indices]

    model = NeuralNetwork(
        learning_rate=1e-1,
        loss_function=rmse,
        lr_decay_rate=0.1,
        lr_decay_fn=learning_rate_staircase_decay,
        lr_decay_steps=500,
        momentum=0.9,
    )
    model.add_layer(
        [
            LayerFactory.dense_layer(input_dim, 10, 'sigmoid'),
            LayerFactory.dense_layer(10, output_dim, 'linear'),
        ]
    )

    model.fit(
        x_shuffled,
        y_shuffled,
        epochs=n_epochs,
        batch_size=batch_size,
        verbose=50,
    )

    # compute r-squared score
    y_pred = model.predict(x_test_std)
    y_train_pred = model.predict(x_train_std)
    r2_result = r2_score(y_train, y_train_pred)

    if plot_result:
        # plot train or test curve
        plt.scatter(x_train, y_train)

        if curve_plot == 'train_curve':
            plt.scatter(x_train, model.predict(x_train_std), c='red')

        if curve_plot == 'test_curve':
            plt.scatter(x_test, model.predict(x_test_std), c='red')
            plt.plot(x_test, model.predict(x_test_std), c='green')

        plt.show()  # type: ignore

    # process arguments
    if args.model_path is not None:
        model.save(args.model_path)

    if args.return_best_loss:
        best_loss = rmse(y_train, y_train_pred, derivative=False)
        print('best loss', best_loss)

    if args.return_r2_score:
        print('r2 score', r2_result)
