import matplotlib.pyplot as plt
import numpy as np
from natural_computing import (
    LayerFactory,
    NeuralNetwork,
    BareBonesParticleSwarmOptimization,
    MinMaxScaler,
    pack_network,
    unpack_network,
    RootMeanSquaredErrorForNN,
    fetch_file_and_convert_to_array,
)


# experiment settings
plot_result = False
curve_plot = 'test_curve'  # train_curve or test_curve
n_iterations = 10000
data_path = (
    'https://raw.githubusercontent.com/gsoaresbaptista/'
    'natural-computing/main/data/regression'
)


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
    x_train = fetch_file_and_convert_to_array(f'{data_path}/x_train.txt')
    y_train = fetch_file_and_convert_to_array(f'{data_path}/y_train.txt')
    x_test = fetch_file_and_convert_to_array(f'{data_path}/x_test.txt')

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
    rmse = RootMeanSquaredErrorForNN(
        x_train_shuffled, y_train_shuffled, decode_guide, 1e-4
    )
    bbpso = BareBonesParticleSwarmOptimization(
        20, n_iterations, [[-1.0, 1.0] for _ in range(individual_shape)]
    )

    bbpso.optimize(rmse)

    # get best model
    best_weights = np.array(bbpso.best_global_position).reshape(-1, 1)
    model = pack_network(best_weights, decode_guide)

    # plot curves
    if plot_result:
        plt.scatter(x_train, y_train, c='blue')

        if curve_plot == 'train_curve':
            plt.plot(x_train, model.predict(x_train_std), c='red')

        if curve_plot == 'test_curve':
            plt.scatter(x_test, model.predict(x_test_std), c='green')
            plt.plot(x_test, model.predict(x_test_std), c='red')

        plt.show()
