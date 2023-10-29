from evolutionary_programming.neural_network import encode_neural_network
from evolutionary_programming.objective_function import (
    RootMeanSquaredErrorForNN, R2ScoreForNN
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


class PredictionExperiment:
    def __init__(self, optimization_method: str):
        self._optimization_method_name = optimization_method

        # get dimensions
        module = NeuralNetworkArchitectures.prediction_architecture()
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
            case _:
                raise ValueError(
                    f'Method {self._optimization_method_name} '
                    'is not recognized. Choose "GA" or "PSO".'
                )

    def run(self, dataset: str) -> float:
        # get data from web
        dataset = DatasetsDownloader.prediction(temperature=dataset)
        (x_train, y_train), (x_test, y_test) = dataset['processed']

        # optimize neural networks using RMSE function
        rmse = RootMeanSquaredErrorForNN(
            x_train, y_train, self._decode_guide, PREDICTION_REGULARIZATION)
        self._optimization_method.optimize(10, rmse)
        best_individual = self._optimization_method.best_individual
        best_fitness = self._optimization_method.best_fitness
        history = self._optimization_method.history

        r2 = R2ScoreForNN(
            x_test, y_test, self._decode_guide, PREDICTION_REGULARIZATION)
        r2_score = r2.evaluate(best_individual)

        return (
            r2_score,
            best_fitness,
            best_individual,
            history,
        )
