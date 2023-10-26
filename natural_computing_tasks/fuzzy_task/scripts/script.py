from fuzzy import FuzzyExperiment, ClassifyFuzzy
import numpy as np
import pandas as pd

optimization_methods = [
    "GA",
    "PSO"
]

iterations = 1
universe = np.arange(0, 1.01, 0.01)

experiment_data = {
    "Method": [],
    "Iteration": [],
    "Best Solution": [],
    "Best Fitness": []
}

for method in optimization_methods:
    print(f"Running {method}...")
    for i in range(iterations):
        print(f"\tIteration {i + 1}")
        fuzzy_function = ClassifyFuzzy(universe)
        experiment = FuzzyExperiment(universe, method, fuzzy_function)
        best_solution = experiment.run()
        best_fitness = fuzzy_function.evaluate(best_solution)
        experiment_data["Method"].append(method)
        experiment_data["Iteration"].append(i + 1)
        experiment_data["Best Solution"].append(best_solution)
        experiment_data["Best Fitness"].append(best_fitness)


df_results = pd.DataFrame(experiment_data)
df_results.to_csv("../analysis/results/fuzzy_task.csv")
print(df_results)