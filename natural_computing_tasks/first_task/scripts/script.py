import os
import subprocess
import csv

experiment_type = [
    'regression',
    'prediction',
    'baseline_prediction',
    'baseline_regression',
]
alternative_approach = [True, True, False, False]
n_exp = 20


# r2 for regression is computed with train data
csv_head = ['experiment_type', 'iteration', 'best_rmse', 'r2_score', 'model_path']
outputs_path = '/home/gabriel/Documents/git/natural-computing-tasks/outputs'
csv_file_path = f'{outputs_path}/results.csv'
rows = []


def run(exp_type, approach, i, alternative):
    model_path = f'{outputs_path}/{exp_type}/models/model_{approach}_{i+1}.pkl'
    print(
        f'- Running {exp_type}.py, experiment number {i + 1} '
        f'for approach {approach}'
    )
    command = (
        f'python {exp_type}.py '
        f'--model-path {model_path}'
        ' --return-best-loss --return-r2-score',
        '--alternative-approach' if alternative else '',
    )

    # run command
    cur_exp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    new_row = {}
    new_row['experiment_type'] = f'{approach}_{exp_type}'
    new_row['iteration'] = i + 1

    # read data from the stdout
    for line in cur_exp.stdout:
        line = line.decode().strip()

        if line.startswith('best loss'):
            new_row['best_rmse'] = line.replace('best loss ', '')

        if line.startswith('r2 score'):
            new_row['r2_score'] = line.replace('r2 score ', '')

    # add the path
    new_row['model_path'] = model_path

    # wait for the subprocess to complete
    cur_exp.wait()
    rows.append(new_row)

    # convert dict to lists
    rows_lst = [list(row.values()) for row in rows]
    rows_lst = [csv_head, *rows_lst]

    with open(csv_file_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        for row in rows_lst:
            csv_writer.writerow(row)


for row_id, exp_type in enumerate(experiment_type):
    for i in range(n_exp):
        # create folders
        if not os.path.exists(f'{outputs_path}/{exp_type}'):
            os.makedirs(f'{outputs_path}/{exp_type}/models')

        # create command
        approach = 'bbpso' if exp_type.startswith('regression') else 'pso'
        approach = 'nn' if exp_type.startswith('baseline') else approach
        run(exp_type, approach, i, False)

        # run alternative approach
        if alternative_approach[row_id]:
            # create command
            approach = 'ga'
            run(exp_type, approach, i, True)
