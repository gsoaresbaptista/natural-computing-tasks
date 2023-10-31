import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme()


df = pd.read_csv('./results.csv')


def set_dataset(x):
    if x.split('_')[-1] == 'min':
        return 'min'
    elif x.split('_')[-1] == 'max':
        return 'max'

    return 'regression'


fig, axs = plt.subplots(1, 3, figsize=(20, 5))

df['Dataset'] = df['Method'].apply(set_dataset)
df['Method_str'] = df['Method'].apply(
    lambda x: x.split('_')[1].capitalize()
    if len(x.split('_')[1]) > 3
    else x.split('_')[1].upper()
)

df_regression = df[df['Dataset'] == 'regression']
df_min = df[df['Dataset'] == 'min']
df_max = df[df['Dataset'] == 'max']

sns.boxplot(x=df_regression['Method_str'],
            y=df['Best Fitness'], ax=axs[0])
axs[0].set(title='Regressão', ylabel='RMSE', xlabel='Método')

# Prediction MIN
sns.boxplot(x=df_min['Method_str'],
            y=df['Best Fitness'], ax=axs[1])
plt.title('Predição (Temperatura Mínima) - Boxplot')
axs[1].set(title='Predição (Temperatura Mínima)', ylabel='RMSE', xlabel='Método')

# Prediction MAX
sns.boxplot(x=df_max['Method_str'],
            y=df['Best Fitness'], ax=axs[2])
plt.title('Predição (Temperatura Máxima) - Boxplot')
plt.ylabel('RMSE')
plt.xlabel('Método')
axs[2].set(title='Predição (Temperatura Máxima)', ylabel='RMSE', xlabel='Método')
fig.savefig('figures/boxplots.png', bbox_inches='tight')
