import argparse
import matplotlib.pyplot as plt
import numpy as np
from natural_computing import (
    LayerFactory,
    NeuralNetwork,
    pack_network,
    unpack_network,
    create_window,
    split_train_test,
    fetch_file_and_convert_to_array,
    MinMaxScaler,
    ParticleSwarmOptimization,
    RootMeanSquaredErrorForNN,
    r2_score,
)


# experiment settings
plot_result = False
curve_plot = 'test_curve'  # train_curve or test_curve
window_size = 7
n_iterations = 10
data_path = (
    'https://raw.githubusercontent.com/gsoaresbaptista/'
    'natural-computing/main/data/prediction'
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
    data = fetch_file_and_convert_to_array(
        f'{data_path}/daily-max-temperatures.csv', skiprows=1, usecols=[1]
    ).squeeze()

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
    rmse = RootMeanSquaredErrorForNN(
        x_train_shuffled, y_train_shuffled, decode_guide, 1e-4
    )
    pso = ParticleSwarmOptimization(
        20,
        n_iterations,
        0.8,
        0.5,
        0.5,
        [[-1, 1] for _ in range(individual_shape)],
    )
    pso.optimize(rmse)

    # get best model
    best_weights = np.array(pso.best_global_position).reshape(-1, 1)
    model = pack_network(best_weights, decode_guide)

    # compute r-squared score
    y_pred = model.predict(x_test_std)
    y_train_pred = model.predict(x_train_std)
    r2_result = r2_score(y_test, y_pred)

    # plot curves
    if plot_result:
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
                y_pred,
                c='red',
                label='predicted',
            )

        if curve_plot == 'train_curve':
            plt.plot(
                range(len(y_train)),
                y_train_pred,
                c='red',
                label='predicted',
            )

        plt.legend()
        plt.show()

    # process arguments
    if args.model_path is not None:
        model.save(args.model_path)

    if args.return_best_loss:
        best_loss = pso.best_global_value
        print('best loss', best_loss)

    if args.return_r2_score:
        print('r2 score', r2_result)
