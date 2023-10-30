import os
import io
import gc
import numpy as np
import pandas as pd
from time import perf_counter
from prediction import PredictionExperiment
from regression import RegressionExperiment
from experiments_setup import DecodeGuides
from evolutionary_programming.neural_network import decode_neural_network
from contextlib import redirect_stdout


# settings
iterations = 10
optimization_methods = ['BACKPROPAGATION', 'GA', 'PSO']
outputs_path = '/home/gabriel/Documents/git/natural-computing-tasks/outputs'


# used data structures
experiment_data = {
    'Method': [],
    'Iteration': [],
    'R2 Score': [],
    'Best Fitness': [],
    'Elapsed Time': [],
}
pred_decode_guide = DecodeGuides.prediction_guide()
regr_decode_guide = DecodeGuides.regression_guide()


# prediction loop
for problem, datasets in [
    ('prediction', ['min', 'max']),
    ('regression', ['outlier']),
]:
    for method in optimization_methods:
        for dataset in datasets:
            for i in range(iterations):

                print(
                    f'\t- iteration {i + 1} with method '
                    f'{method} for {problem} problem'
                )

                # block stdout prints
                output_buffer = io.StringIO()
                with redirect_stdout(output_buffer):

                    # get experiment class
                    if problem == 'prediction':
                        experiment_class = PredictionExperiment
                        decode_guide = pred_decode_guide
                    elif problem == 'regression':
                        experiment_class = RegressionExperiment
                        decode_guide = regr_decode_guide
                    else:
                        raise ValueError(f'Problem {problem} is unknown')

                    experiment = experiment_class(method)
                    start_time = perf_counter()
                    r2, fitness, individual, history = experiment.run(dataset)
                    end_time = perf_counter()

                # save info into dict
                experiment_data['Iteration'].append(i + 1)
                experiment_data['R2 Score'].append(r2)
                experiment_data['Best Fitness'].append(fitness)
                experiment_data['Elapsed Time'].append(end_time - start_time)
                experiment_data['Method'].append(
                    f'{problem}_{method}_{dataset}')

                # save artifacts
                folder_path = f'{outputs_path}/{problem}/{method}_{dataset}'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                module = decode_neural_network(individual, decode_guide)
                module.save(f'{folder_path}/model_{i+1}.pkl')
                np.save(
                    f'{folder_path}/history_{i+1}.npy',
                    np.array(history),
                    allow_pickle=True,
                )

                # save csv
                if not os.path.exists(outputs_path):
                    os.makedirs(outputs_path)
                pd.DataFrame(experiment_data).to_csv(
                    f'{outputs_path}/results.csv', index=False
                )

                gc.collect()
