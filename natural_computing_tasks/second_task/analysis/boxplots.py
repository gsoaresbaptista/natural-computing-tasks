import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme()


df = pd.read_csv('./results.csv')


def set_dataset(x):
    return x.split('_')[-1]


fig, axs = plt.subplots(1, 3, figsize=(20, 5))

df['Dataset'] = df['Method'].apply(set_dataset)
df['Method_str'] = df['Method'].apply(
    lambda x: x.split('_')[1].capitalize()
    if len(x.split('_')[1]) > 3
    else x.split('_')[1].upper()
)

df_iris = df[df['Dataset'] == 'iris']
df_heart = df[df['Dataset'] == 'heart']
df_hepatitis = df[df['Dataset'] == 'hepatitis']

sns.boxplot(x=df_iris['Method_str'],
            y=df_iris['Accuracy'], ax=axs[0])
axs[0].set(title='Iris', ylabel='Fitness', xlabel='Método')

sns.boxplot(x=df_heart['Method_str'],
            y=df_heart['Accuracy'], ax=axs[1])
axs[1].set(title='Heart', ylabel='Fitness', xlabel='Método')

sns.boxplot(x=df_hepatitis['Method_str'],
            y=df_hepatitis['Accuracy'], ax=axs[2])
axs[2].set(title='Hepatitis', ylabel='Fitness', xlabel='Método')

fig.savefig('figures/boxplots.png', bbox_inches='tight')
