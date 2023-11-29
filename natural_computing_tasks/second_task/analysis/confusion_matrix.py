import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from evolutionary_programming.neural_network import NeuralNetwork
from experiments_setup import DatasetsDownloader
from sklearn.metrics import confusion_matrix

np.random.seed(42)
sns.set_theme()


def model_pred(model, x, shape) -> np.ndarray:
    def softmax(logits):
        exp_logits = np.exp(logits)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return softmax_probs

    y_pred = model.predict(x)
    logits = np.argmax(softmax(y_pred), axis=1)
    y_pred = np.zeros(shape=shape)

    for i, logit in enumerate(logits):
        y_pred[i, logit] = 1

    return y_pred


datasets = {
    "iris": DatasetsDownloader.iris()["processed"],
    "heart": DatasetsDownloader.heart()["processed"],
    "hepatitis": DatasetsDownloader.hepatitis()["processed"],
}


for dataset in ["iris", "heart", "hepatitis"]:
    (_, _), (x, y) = datasets[dataset]
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

    model = NeuralNetwork.load(f"best_models/backpropagation_{dataset}.pkl")
    heart_pred = model_pred(model, x, shape=y.shape)
    matrix = confusion_matrix(np.argmax(heart_pred, axis=1), np.argmax(y, axis=1))
    sns.heatmap(matrix, annot=True, ax=axs[0], cmap="Blues")
    axs[0].set(title="Backpropagation", ylabel="Verdadeiro", xlabel="Predito")

    model = NeuralNetwork.load(f"best_models/ga_{dataset}.pkl")
    heart_pred = model_pred(model, x, shape=y.shape)
    matrix = confusion_matrix(np.argmax(heart_pred, axis=1), np.argmax(y, axis=1))
    sns.heatmap(matrix, annot=True, ax=axs[1], cmap="Blues")
    axs[1].set(title="GA", ylabel="Verdadeiro", xlabel="Predito")

    model = NeuralNetwork.load(f"best_models/pso_{dataset}.pkl")
    heart_pred = model_pred(model, x, shape=y.shape)
    matrix = confusion_matrix(np.argmax(heart_pred, axis=1), np.argmax(y, axis=1))
    sns.heatmap(matrix, annot=True, ax=axs[2], cmap="Blues")
    axs[2].set(title="PSO", ylabel="Verdadeiro", xlabel="Predito")

    fig.savefig(f"figures/cm_{dataset}.png", bbox_inches="tight")
