import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from evolutionary_programming.neural_network import NeuralNetwork
from evolutionary_programming.data_processing import (
    fetch_csv_to_numpy, create_window, split_train_test, MinMaxScaler)

sns.set_theme()

regression_train_url = (
    'https://raw.githubusercontent.com/gsoaresbaptista/'
    'datasets/main/datasets/outlier_regression/train.csv'
)

# plot regression curve
columns, x, y = fetch_csv_to_numpy(regression_train_url)
x, y = x.squeeze(), y.squeeze()
sns.scatterplot(x=x, y=y, linewidth=0)
plt.title('Pontos de treinamento - Regressão')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('figures/regression_data.png', bbox_inches='tight')
plt.cla()

# get base temperatures
temperatures_url = (
    'https://raw.githubusercontent.com/gsoaresbaptista/'
    'datasets/main/datasets/temperatures.csv'
)
columns, min_temps, max_temps = fetch_csv_to_numpy(
    temperatures_url, columns=[1, 2])

# min temperatures
(x_train, y_train), (x_test, y_test) = split_train_test(
    np.array(range(min_temps.shape[0])),
    min_temps.squeeze(),
    train_percentage=0.8, sequential=True
)

plt.figure(figsize=(15, 4))
sns.lineplot(x=x_train, y=y_train, label='Treino')
sns.lineplot(x=x_test, y=y_test, label='Teste')
plt.title('Conjunto de dados - Predição (Temperatura Mínima)')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('figures/pred_data_min.png', bbox_inches='tight')
plt.cla()

# max temperatures
(x_train, y_train), (x_test, y_test) = split_train_test(
    np.array(range(max_temps.shape[0])),
    max_temps.squeeze(),
    train_percentage=0.8, sequential=True
)

plt.figure(figsize=(15, 4))
sns.lineplot(x=x_train, y=y_train, label='Treino')
sns.lineplot(x=x_test, y=y_test, label='Teste')
plt.title('Conjunto de dados - Predição (Temperatura Máxima)')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('figures/pred_data_max.png', bbox_inches='tight')
