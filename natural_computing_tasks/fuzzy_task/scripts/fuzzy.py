from natural_computing import (
    MinMaxScaler,
    RealGeneticAlgorithm,
    BaseFunction,
    BareBonesParticleSwarmOptimization
)

from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import skfuzzy as fuzz
from typing import List
import numpy as np

class ClassifyFuzzy(BaseFunction):
    def __init__(self, universe: List[np.ndarray]):
        self.universe = universe
        self.iris = datasets.load_iris()
        self.y = self.iris.target
        self.x = self.iris.data

        scaler = MinMaxScaler()
        scaler.fit(self.x)

        self.x = scaler.transform(self.x)

    def evaluate(self, point: List[float]) -> float:
        # punishes the points that are out of the search space
        if any(p > 1 or p < 0 for p in point):
            return 1
        
        # sorts to w1 < w2 < w3 < w4
        point.sort()

        w1, w2, w3, w4 = point
        
        #Definindo as funções de pertinencia dos antecedentes
        x1_short = fuzz.trimf(self.universe, [0., 0., w1])
        x1_middle = fuzz.trimf(self.universe, [0.,w1, 1.0])
        x1_long = fuzz.trimf(self.universe, [w1, 1.0, 1.0])

        x2_short = fuzz.trimf(self.universe, [0., 0., w2])
        x2_middle = fuzz.trimf(self.universe, [0.,w2, 1.0])
        x2_long = fuzz.trimf(self.universe, [w2, 1.0, 1.0])

        x3_short = fuzz.trimf(self.universe, [0., 0., w3])
        x3_middle = fuzz.trimf(self.universe, [0.,w3, 1.0])
        x3_long = fuzz.trimf(self.universe, [w3, 1.0, 1.0])

        x4_short = fuzz.trimf(self.universe, [0., 0., w4])
        x4_middle = fuzz.trimf(self.universe, [0.,w4, 1.0])
        x4_long = fuzz.trimf(self.universe, [w4, 1.0, 1.0])

        x1_short_x = fuzz.interp_membership(self.universe, x1_short, self.x[:,0])
        x1_middle_x = fuzz.interp_membership(self.universe, x1_middle, self.x[:,0])
        x1_long_x = fuzz.interp_membership(self.universe, x1_long, self.x[:,0])

        x2_short_x = fuzz.interp_membership(self.universe, x2_short, self.x[:,1])
        x2_middle_x = fuzz.interp_membership(self.universe, x2_middle, self.x[:,1])
        x2_long_x = fuzz.interp_membership(self.universe, x2_long, self.x[:,1])

        x3_short_x = fuzz.interp_membership(self.universe, x3_short, self.x[:,2])
        x3_middle_x = fuzz.interp_membership(self.universe, x3_middle, self.x[:,2])
        x3_long_x = fuzz.interp_membership(self.universe, x3_long, self.x[:,2])

        x4_short_x = fuzz.interp_membership(self.universe, x4_short, self.x[:,3])
        x4_middle_x = fuzz.interp_membership(self.universe, x4_middle, self.x[:,3])
        x4_long_x = fuzz.interp_membership(self.universe, x4_long, self.x[:,3])

        # Aplica as regras fuzzy
        setosa = np.fmin(np.fmax(x3_short_x , x3_middle_x), x4_short_x )

        versicolor = np.fmax(np.fmin(np.fmin(np.fmin(np.fmax(x1_short_x , x1_long_x), 
                    np.fmax(x2_middle_x, x2_long_x )), np.fmax(x3_middle_x, x3_long_x)),x4_middle_x), 
                    np.fmin(x1_middle_x, np.fmin(np.fmin(np.fmax(x2_long_x , x2_middle_x),x3_short_x ), x4_long_x)))

        virginica = np.fmin(np.fmin(np.fmax(x2_long_x , x2_middle_x), x3_long_x), x4_long_x)

        y_ = np.argmax([setosa, versicolor, virginica], axis=0)

        result = (self.y == y_).mean()

        # 1 - result to minimize instead of maximize
        return 1 - result


if __name__ == "__main__":
    universe = np.arange(0, 1.01, 0.01)
    fuzzy_function = ClassifyFuzzy(universe)
    
    ga = RealGeneticAlgorithm(
        100, 1000, [[0.0, 1.0] for _ in range(4)]
    )

    ga.optimize(fuzzy_function)

    ga_best = ga.best_global_phenotype
    ga_accuracy = (1 - fuzzy_function.evaluate(ga_best)) * 100
    print("----Real Genetic Algorithm----")
    print(f"The best phenotype ({ga_best}) produced the solution with accuracy: {ga_accuracy:.2f}%")
    

    pso = BareBonesParticleSwarmOptimization(80, 1000, [[0.0, 1.0] for _ in range(4)])
    pso.optimize(fuzzy_function)

    pso_best = pso.best_global_position
    pso_accuracy = (1 - fuzzy_function.evaluate(pso_best)) * 100
    print("----Bare Bones Particle Swarm Optimization----")
    print(f"The best position ({pso_best}) produced the solution with accuracy: {pso_accuracy:.2f}%")