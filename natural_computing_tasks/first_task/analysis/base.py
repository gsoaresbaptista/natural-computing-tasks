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
