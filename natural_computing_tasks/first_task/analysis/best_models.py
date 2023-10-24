import matplotlib.pyplot as plt
from natural_computing import (
    NeuralNetwork,
    MinMaxScaler,
    create_window,
    split_train_test,
    fetch_file_and_convert_to_array,
)

data_path = (
    'https://raw.githubusercontent.com/gsoaresbaptista/'
    'natural-computing/main/data'
)

# pred min figure
data = fetch_file_and_convert_to_array(
    f'{data_path}/prediction/daily-min-temperatures.csv',
    skiprows=1, usecols=[1],
).squeeze()

x, y = create_window(data, window_size=7)
(x_train, y_train), (x_test, y_test) = split_train_test(
    x, y, 0.8, sequential=True
)
scaler = MinMaxScaler(centered_on_zero=False)
scaler.fit(x_train)
x_train_std = scaler.transform(x_train)
x_test_std = scaler.transform(x_test)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, method in enumerate(['pso', 'ga', 'nn']):
    model = NeuralNetwork.load(f'best_models/pred_min/model_{method}.pkl')
    y_pred = model.predict(x_test_std)
    y_train_pred = model.predict(x_train_std)
    axs[i].plot(
        range(len(y_train), len(y_train) + len(y_test)),
        y_test,
        c='blue',
        label='real',
    )
    axs[i].plot(
        range(len(y_train), len(y_train) + len(y_test)),
        y_pred,
        c='red',
        label='predicted',
    )
    method = 'backpropagation' if method == 'nn' else method
    method = method.upper() if method in ['pso', 'ga'] else method.capitalize()
    axs[i].set_title(method)
plt.show()

# reg figure
x_train = fetch_file_and_convert_to_array(f'{data_path}/regression/x_train.txt')
y_train = fetch_file_and_convert_to_array(f'{data_path}/regression/y_train.txt')
x_test = fetch_file_and_convert_to_array(f'{data_path}/regression/x_test.txt')
scaler = MinMaxScaler(centered_on_zero=False)
scaler.fit(x_train)
x_train_std = scaler.transform(x_train)
x_test_std = scaler.transform(x_test)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, method in enumerate(['pso', 'ga', 'nn']):
    model = NeuralNetwork.load(f'best_models/reg/model_{method}.pkl')
    y_pred = model.predict(x_test_std)
    y_train_pred = model.predict(x_train_std)
    axs[i].scatter(x_train, y_train)
    axs[i].scatter(x_test, model.predict(x_test_std), c='red')
    axs[i].plot(x_test, model.predict(x_test_std), c='green')
    method = 'backpropagation' if method == 'nn' else method
    method = method.upper() if method in ['pso', 'ga'] else method.capitalize()
    axs[i].set_title(method)
plt.show()
