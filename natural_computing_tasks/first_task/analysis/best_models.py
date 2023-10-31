import matplotlib.pyplot as plt
import seaborn as sns
from evolutionary_programming.neural_network import NeuralNetwork
from evolutionary_programming.data_processing import (
    fetch_csv_to_numpy, create_window, split_train_test, MinMaxScaler)

sns.set_theme()

temperatures_url = (
    'https://raw.githubusercontent.com/gsoaresbaptista/'
    'datasets/main/datasets/temperatures.csv'
)
regression_train_url = (
    'https://raw.githubusercontent.com/gsoaresbaptista/'
    'datasets/main/datasets/outlier_regression/train.csv'
)
regression_test_url = (
    'https://raw.githubusercontent.com/gsoaresbaptista/'
    'datasets/main/datasets/outlier_regression/test.csv'
)

# get temperature data
columns, min_temps, max_temps = fetch_csv_to_numpy(
    temperatures_url, columns=[1, 2])

# pred min figure
x, y = create_window(min_temps, window_size=3)
(x_train, y_train), (x_test, y_test) = split_train_test(
    x, y, 0.8, sequential=True)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_std = scaler.transform(x_train)
x_test_std = scaler.transform(x_test)

# min temperatures
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, method in enumerate(['pso', 'ga', 'backpropagation']):
    model = NeuralNetwork.load(f'best_models/prediction/{method}_min.pkl')
    y_pred = model.predict(x_test_std)
    axs[i].plot(
        range(len(y_train), len(y_train) + len(y_test)),
        y_test,
        c='blue',
        label='Real',
    )
    axs[i].plot(
        range(len(y_train), len(y_train) + len(y_test)),
        scaler.inverse_transform(y_pred),
        c='red',
        label='Predito',
    )
    method = method.upper() if method in ['pso', 'ga'] else method.capitalize()
    axs[i].set_title(method)
    axs[i].set(xlabel='X', ylabel='Temperatura')
    axs[i].legend()
fig.suptitle('Predição (Temperatura Mínima) - Conjunto de Teste')
fig.savefig('figures/pred_min_temp_curves.png', bbox_inches='tight')

# max temperatures
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, method in enumerate(['pso', 'ga', 'backpropagation']):
    model = NeuralNetwork.load(f'best_models/prediction/{method}_max.pkl')
    y_pred = model.predict(x_test_std)
    axs[i].plot(
        range(len(y_train), len(y_train) + len(y_test)),
        y_test,
        c='blue',
        label='Real',
    )
    axs[i].plot(
        range(len(y_train), len(y_train) + len(y_test)),
        scaler.inverse_transform(y_pred),
        c='red',
        label='Predito',
    )
    method = method.upper() if method in ['pso', 'ga'] else method.capitalize()
    axs[i].set_title(method)
    axs[i].set(xlabel='X', ylabel='Temperatura')
    axs[i].legend()
fig.suptitle('Predição (Temperatura Máxima) - Conjunto de Teste')
fig.savefig('figures/pred_max_temp_curves.png', bbox_inches='tight')

# get regression data
columns, x_train, y_train = fetch_csv_to_numpy(
    regression_train_url)
columns, x_test = fetch_csv_to_numpy(
    regression_test_url)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# regression
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, method in enumerate(['pso', 'ga', 'backpropagation']):
    model = NeuralNetwork.load(f'best_models/regression/{method}.pkl')
    y_pred = model.predict(x_test)
    axs[i].scatter(
        x_train,
        y_train,
        c='blue',
        label='Real',
    )
    axs[i].scatter(
        x_test,
        y_pred,
        c='red',
        label='Predito',
    )
    method = method.upper() if method in ['pso', 'ga'] else method.capitalize()
    axs[i].set_title(method)
    axs[i].set(xlabel='X', ylabel='Y')
    axs[i].legend()
fig.suptitle('Regressão')
fig.savefig('figures/regression_curves.png', bbox_inches='tight')
