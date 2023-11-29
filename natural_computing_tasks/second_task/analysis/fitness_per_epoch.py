import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# iris
ga = np.load('best_histories/ga_iris.npy')
pso = np.load('best_histories/pso_iris.npy')

last_result, last_x = float('inf'), 0

for i in range(len(ga)):
    if ga[i] != last_result:
        last_result, last_x = ga[i], i

x = np.arange(0, last_x)
sns.lineplot(x=x, y=pso[:last_x], label='PSO')
sns.lineplot(x=x, y=ga[:last_x], label='GA')
plt.title('Fitness por Iteração - Iris')
plt.xlabel('Iteração')
plt.ylabel('Fitness')
plt.savefig('figures/iris.png', bbox_inches='tight')
plt.cla()


# heart
ga = np.load('best_histories/ga_heart.npy')
pso = np.load('best_histories/history_10.npy')

print(ga)
print(pso)

last_result, last_x = float('inf'), 0

for i in range(len(ga)):
    if ga[i] != last_result:
        last_result, last_x = ga[i], i

x = np.arange(0, last_x)
sns.lineplot(x=x, y=pso[:last_x], label='PSO')
sns.lineplot(x=x, y=ga[:last_x], label='GA')
plt.title('Fitness por Iteração - Iris')
plt.xlabel('Iteração')
plt.ylabel('Fitness')
plt.savefig('figures/heart.png', bbox_inches='tight')
plt.cla()


# heart
ga = np.load('best_histories/ga_hepatitis.npy')
pso = np.load('best_histories/pso_hepatitis.npy')

last_result, last_x = float('inf'), 0

for i in range(len(ga)):
    if ga[i] != last_result:
        last_result, last_x = ga[i], i

x = np.arange(0, last_x)
sns.lineplot(x=x, y=pso[:last_x], label='PSO')
sns.lineplot(x=x, y=ga[:last_x], label='GA')
plt.title('Fitness por Iteração - Iris')
plt.xlabel('Iteração')
plt.ylabel('Fitness')
plt.savefig('figures/hepatitis.png', bbox_inches='tight')
plt.cla()
