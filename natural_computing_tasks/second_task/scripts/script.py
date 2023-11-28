import os
import io
import gc
import numpy as np
import pandas as pd
from time import perf_counter
from heart import HeartExperiment
from hepatitis import HepatitisExperiment
from iris import IrisExperiment
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
    'Accuracy': [],
    'Best Fitness': [],
    'Elapsed Time': [],
}

# prediction loop
for problem, datasets in [
    ('classification', ['iris', 'heart', 'hepatitis']),
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
                    if dataset == 'iris':
                        experiment_class = IrisExperiment
                        decode_guide = DecodeGuides.iris_guide()
                    elif dataset == 'heart':
                        experiment_class = HeartExperiment
                        decode_guide = DecodeGuides.heart_guide()
                    elif dataset == 'hepatitis':
                        experiment_class = HepatitisExperiment
                        decode_guide = DecodeGuides.hepatitis_guide()
                    else:
                        raise ValueError(f'Problem {problem} is unknown')

                    experiment = experiment_class(method)
                    start_time = perf_counter()
                    acc, fitness, individual, history = experiment.run()
                    end_time = perf_counter()

                # save info into dict
                experiment_data['Iteration'].append(i + 1)
                experiment_data['Accuracy'].append(acc)
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
