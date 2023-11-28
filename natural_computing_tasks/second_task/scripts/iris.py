from utils import accuracy_test
from evolutionary_programming.neural_network import encode_neural_network, decode_neural_network
from evolutionary_programming.objective_function import (
    AccuracyErrorForNN
)
from evolutionary_programming.optimization import (
    GeneticAlgorithm,
    ParticleSwarm,
    PopulationBasedOptimizer,
)

from experiments_setup import (
    DatasetsDownloader, NeuralNetworkArchitectures,
    PREDICTION_REGULARIZATION
)


class IrisExperiment:
    def __init__(self, optimization_method: str):
        self._optimization_method_name = optimization_method
        self._backpropagation = False

        # get dimensions
        module = NeuralNetworkArchitectures.iris_architecture()
        individual, self._decode_guide = encode_neural_network(module)
        self._dimensions = individual.shape[0]
        self._optimization_method = self._init_optimization_method()

    def _init_optimization_method(self) -> PopulationBasedOptimizer:
        # create bounds
        min_bounds = [-1 for _ in range(self._dimensions)]
        max_bounds = [+1 for _ in range(self._dimensions)]

        match self._optimization_method_name:
            case 'GA':
                return GeneticAlgorithm(
                    100,
                    self._dimensions,
                    min_bounds,
                    max_bounds,
                    elitist_individuals=10,
                    mutation_probability=0.05,
                    bounded=False,
                )
            case 'PSO':
                return ParticleSwarm(
                    30,
                    self._dimensions,
                    min_bounds,
                    max_bounds,
                    cognitive=0.75,
                    social=0.25,
                    inertia=0.8,
                    max_stagnation_interval=5,
                    bounded=False,
                )
            case 'BACKPROPAGATION':
                self._backpropagation = True
                return ()
            case _:
                raise ValueError(
                    f'Method {self._optimization_method_name} '
                    'is not recognized. Choose "BACKPROPAGATION",'
                    ' "GA" or "PSO".'
                )

    def run(self) -> float:
        # get data from web
        dataset = DatasetsDownloader.iris()
        (x_train, y_train), (x_test, y_test) = dataset['processed']

        if self._backpropagation:
            # optimize neural networks using backpropagation algorithm
            model = NeuralNetworkArchitectures.iris_architecture()
            model.fit(x_train, y_train, epochs=15000)

            # get data
            best_fitness = model.evaluate(x_test, y_test)
            best_individual, _ = encode_neural_network(model)
            history = []
        else:
            # optimize neural networks using RMSE function
            acc = AccuracyErrorForNN(
                x_train, y_train, self._decode_guide,
                PREDICTION_REGULARIZATION)
            self._optimization_method.optimize(2000, acc)

            # get data
            best_individual = self._optimization_method.best_individual
            best_fitness = self._optimization_method.best_fitness
            history = self._optimization_method.history

            model = decode_neural_network(best_individual, self._decode_guide)

        acc = accuracy_test(model, x_test, y_test)

        return (
            acc,
            best_fitness,
            best_individual,
            history,
        )


if __name__ == '__main__':
    # example of use
    exp = IrisExperiment('BACKPROPAGATION')
    acc, bf, best_individual, _ = exp.run()
    print(acc)
