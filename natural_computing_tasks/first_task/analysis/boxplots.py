import pandas as pd
import matplotlib.pyplot as plt

df = pd.concat([pd.read_csv(f'results/results{i + 1}.csv') for i in range(3)])

df_grouped = df.groupby('experiment_type').agg(
    {'best_rmse': ['mean', 'std'], 'r2_score': ['mean', 'std']}
)

# Create a boxplot with the three experiment types
fig, ax = plt.subplots(1, 3, figsize=(20, 5))

df_regression = df[df['experiment_type'].isin(['bbpso_regression', 'nn_baseline_regression', 'ga_regression'])]
df_regression = df_regression.replace({'bbpso_regression': 'PSO', 'nn_baseline_regression': 'Backpropagation', 'ga_regression': 'GA'})
df_regression.boxplot(column=['best_rmse'], by='experiment_type', ax=ax[0])
ax[0].set_title('Regression')
ax[0].set_ylabel('Best RMSE - Regression')
ax[0].set_xlabel('')

df_predict_min_temp = df[df['experiment_type'].isin(['pso_prediction_min_temperature', 'nn_baseline_prediction_min_temperature', 'ga_prediction_min_temperature'])]
df_predict_min_temp = df_predict_min_temp.replace({'pso_prediction_min_temperature': 'PSO', 'nn_baseline_prediction_min_temperature': 'Backpropagation', 'ga_prediction_min_temperature': 'GA'})
df_predict_min_temp.boxplot(column=['best_rmse'], by='experiment_type', ax=ax[1])
ax[1].set_title('Prediction Min Temperature')
ax[1].set_ylabel('Best RMSE - Prediction Min Temperature')
ax[1].set_xlabel('')

df_predict_max_temp = df[df['experiment_type'].isin(['pso_prediction_max_temperature', 'nn_baseline_prediction_max_temperature', 'ga_prediction_max_temperature'])]
df_predict_max_temp = df_predict_max_temp.replace({'pso_prediction_max_temperature': 'PSO', 'nn_baseline_prediction_max_temperature': 'Backpropagation', 'ga_prediction_max_temperature': 'GA'})
df_predict_max_temp.boxplot(column=['best_rmse'], by='experiment_type', ax=ax[2])
ax[2].set_title('Prediction Max Temperature')
ax[2].set_ylabel('Best RMSE - Prediction Max Temperature')
ax[2].set_xlabel('')

plt.suptitle('')
plt.show()
