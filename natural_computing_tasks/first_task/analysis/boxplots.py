import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

df = pd.read_csv('./results.csv')

def set_dataset(x):
    if x.split('_')[-1]== 'min':
        return 'min'
    elif x.split('_')[-1] == 'max':
        return 'max'
    
    return 'regression'

df['Dataset'] = df['Method'].apply(set_dataset)
df['Method_str'] = df['Method'].apply(lambda x: x.split('_')[1].capitalize() if len(x.split('_')[1]) > 3 else x.split('_')[1].upper())

df_regression = df[df['Dataset'] == 'regression']
df_min = df[df['Dataset'] == 'min']
df_max = df[df['Dataset'] == 'max']

sns.boxplot(x = df_regression['Method_str'], 
            y = df['Best Fitness'])
plt.title('Regressão - Boxplot')
plt.ylabel('RMSE')
plt.xlabel('Método')
plt.savefig('./figures/regression_boxplot.png', bbox_inches='tight')
plt.cla()

#Prediction MIN

sns.boxplot(x = df_min['Method_str'], 
            y = df['Best Fitness'])
plt.title('Predição (Temperatura Mínima) - Boxplot')
plt.ylabel('RMSE')
plt.xlabel('Método')
plt.savefig('./figures/prediction_min_boxplot.png', bbox_inches='tight')
plt.cla()

#Prediction MAX

sns.boxplot(x = df_max['Method_str'], 
            y = df['Best Fitness'])
plt.title('Predição (Temperatura Máxima) - Boxplot')
plt.ylabel('RMSE')
plt.xlabel('Método')
plt.savefig('./figures/prediction_max_boxplot.png', bbox_inches='tight')
plt.cla()

print(df)
