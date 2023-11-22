import numpy as np
from sklearn.metrics import accuracy_score
from evolutionary_programming.neural_network import (
    NeuralNetwork,
)

def accuracy_test(model: NeuralNetwork, x_test: np.ndarray, y_test: np.ndarray) -> float:
    y_pred = model.predict(x_test)

    def softmax(logits):
        exp_logits = np.exp(logits)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return softmax_probs

    y_pred = np.argmax(softmax(y_pred), axis=1)
    y_real = np.argmax(y_test, axis=1)

    return accuracy_score(y_real, y_pred)
