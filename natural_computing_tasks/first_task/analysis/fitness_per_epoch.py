import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# prediction, min temperature
pso = np.load('best_histories/prediction/pso_min.npy')
ga = np.load('best_histories/prediction/ga_min.npy')

last_result, last_x = float('inf'), 0

for i in range(len(pso)):
    if pso[i] != last_result:
        last_result, last_x = pso[i], i

x = np.arange(0, last_x)
sns.lineplot(x=x, y=pso[:last_x], label='PSO')
sns.lineplot(x=x, y=ga[:last_x], label='GA')
plt.title('RMSE por Iteração - Predição (Temperatura Mínima)')
plt.xlabel('Iteração')
plt.ylabel('RMSE')
plt.savefig('figures/pred_min_temp.png', bbox_inches='tight')
plt.cla()

# prediction, max temperature
pso = np.load('best_histories/prediction/pso_max.npy')
ga = np.load('best_histories/prediction/ga_max.npy')

last_result, last_x = float('inf'), 0

for i in range(len(pso)):
    if pso[i] != last_result:
        last_result, last_x = pso[i], i

x = np.arange(0, last_x)
sns.lineplot(x=x, y=pso[:last_x], label='PSO')
sns.lineplot(x=x, y=ga[:last_x], label='GA')
plt.title('RMSE por Iteração - Predição (Temperatura Máxima)')
plt.xlabel('Iteração')
plt.ylabel('RMSE')
plt.savefig('figures/pred_max_temp.png', bbox_inches='tight')
plt.cla()

# regression
pso = np.load('best_histories/regression/pso.npy')
ga = np.load('best_histories/regression/ga.npy')

last_result, last_x = float('inf'), 0

for i in range(len(pso)):
    if pso[i] != last_result:
        last_result, last_x = pso[i], i

x = np.arange(0, last_x)
sns.lineplot(x=x, y=pso[:last_x], label='PSO')
sns.lineplot(x=x, y=ga[:last_x], label='GA')
plt.title('RMSE por Iteração - Regressão')
plt.xlabel('Iteração')
plt.ylabel('RMSE')
plt.savefig('figures/regression.png', bbox_inches='tight')
