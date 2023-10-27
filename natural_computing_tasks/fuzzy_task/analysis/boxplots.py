import pandas as pd
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import ast

df = pd.read_csv('./results/fuzzy_task.csv')

df_pso = df[df['Method'] == 'PSO']
df_ga = df[df['Method'] == 'GA']

# Get the index of the row with the highest value in the 'column_name' column
highest_value_index_pso = df_pso['Best Fitness'].idxmin()
highest_value_index_ga = df_ga['Best Fitness'].idxmin()

# Get the row with the highest value
highest_value_row_pso = df.loc[highest_value_index_pso]
highest_value_index_ga = df.loc[highest_value_index_ga]

print(ast.literal_eval(highest_value_row_pso["Best Solution"]), highest_value_row_pso["Best Fitness"])
print(ast.literal_eval(highest_value_index_ga["Best Solution"]), highest_value_index_ga["Best Fitness"])

def plota_fuzzy_membership_function(W, title):
    iris = datasets.load_iris()
    y = iris.target
    x = iris.data

    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    universe = np.arange(0, 1.01, 0.01)
    w1, w2, w3, w4 = W
    #Definindo as funções de pertinencia dos antecedentes
    x1_short = fuzz.trimf(universe, [0., 0., w1])
    x1_middle = fuzz.trimf(universe, [0.,w1, 1.0])
    x1_long = fuzz.trimf(universe, [w1, 1.0, 1.0])

    x2_short = fuzz.trimf(universe, [0., 0., w2])
    x2_middle = fuzz.trimf(universe, [0.,w2, 1.0])
    x2_long = fuzz.trimf(universe, [w2, 1.0, 1.0])

    x3_short = fuzz.trimf(universe, [0., 0., w3])
    x3_middle = fuzz.trimf(universe, [0.,w3, 1.0])
    x3_long = fuzz.trimf(universe, [w3, 1.0, 1.0])

    x4_short = fuzz.trimf(universe, [0., 0., w4])
    x4_middle = fuzz.trimf(universe, [0.,w4, 1.0])
    x4_long = fuzz.trimf(universe, [w4, 1.0, 1.0])

    # Plota as funções de pertinencia
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(6, 15))

    ax0.plot(universe, x1_short, 'b', linewidth=1.5, label='short')
    ax0.plot(universe, x1_middle, 'g', linewidth=1.5, label='middle')
    ax0.plot(universe, x1_long, 'r', linewidth=1.5, label='long')
    ax0.set_title('Sepal length - x1')
    ax0.legend()

    ax1.plot(universe, x2_short, 'b', linewidth=1.5, label='short')
    ax1.plot(universe, x2_middle, 'g', linewidth=1.5, label='middle')
    ax1.plot(universe, x2_long, 'r', linewidth=1.5, label='long')
    ax1.set_title('Sepal width - x2')
    ax1.legend()

    ax2.plot(universe, x3_short, 'b', linewidth=1.5, label='short')
    ax2.plot(universe, x3_middle, 'g', linewidth=1.5, label='middle')
    ax2.plot(universe, x3_long, 'r', linewidth=1.5, label='long')
    ax2.set_title('Petal length - x3')
    ax2.legend()

    ax3.plot(universe, x4_short, 'b', linewidth=1.5, label='short')
    ax3.plot(universe, x4_middle, 'g', linewidth=1.5, label='middle')
    ax3.plot(universe, x4_long, 'r', linewidth=1.5, label='long')
    ax3.set_title('Petal width - x4')
    ax3.legend()
    fig.show()
    fig.savefig(f'./results/images/{title}.png')

    #Ativa as funções de pertinência com os inputs
    x1_short_x = fuzz.interp_membership(universe, x1_short, x[:,0])
    x1_middle_x = fuzz.interp_membership(universe, x1_middle, x[:,0])
    x1_long_x = fuzz.interp_membership(universe, x1_long, x[:,0])

    x2_short_x = fuzz.interp_membership(universe, x2_short, x[:,1])
    x2_middle_x = fuzz.interp_membership(universe, x2_middle, x[:,1])
    x2_long_x = fuzz.interp_membership(universe, x2_long, x[:,1])

    x3_short_x = fuzz.interp_membership(universe, x3_short, x[:,2])
    x3_middle_x = fuzz.interp_membership(universe, x3_middle, x[:,2])
    x3_long_x = fuzz.interp_membership(universe, x3_long, x[:,2])

    x4_short_x = fuzz.interp_membership(universe, x4_short, x[:,3])
    x4_middle_x = fuzz.interp_membership(universe, x4_middle, x[:,3])
    x4_long_x = fuzz.interp_membership(universe, x4_long, x[:,3])

    # Aplica as regras fuzzy
    setosa = np.fmin(np.fmax(x3_short_x , x3_middle_x), x4_short_x )

    versicolor = np.fmax(np.fmin(np.fmin(np.fmin(np.fmax(x1_short_x , x1_long_x), 
                np.fmax(x2_middle_x, x2_long_x )), np.fmax(x3_middle_x, x3_long_x)),x4_middle_x), 
                np.fmin(x1_middle_x, np.fmin(np.fmin(np.fmax(x2_long_x , x2_middle_x),x3_short_x ), x4_long_x)))

    virginica = np.fmin(np.fmin(np.fmax(x2_long_x , x2_middle_x), x3_long_x), x4_long_x)


    # O maior valor dentre as tres regras vai ser a classe predita 
    y_ = np.argmax([setosa, versicolor, virginica], axis=0)

    fig1, ax4 = plt.subplots()
    conf_mat = confusion_matrix(y,y_)
    df_cm = pd.DataFrame(conf_mat,index =['Setosa', 'Versicolor', 'Virginica'], columns = ['Setosa', 'Versicolor', 'Virginica'])
    sns.heatmap(df_cm, annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig1.savefig(f'./results/images/{title}_confusion_matrix.png')
    fig1.show()
    

plota_fuzzy_membership_function(ast.literal_eval(highest_value_row_pso["Best Solution"]), "PSO")
plota_fuzzy_membership_function(ast.literal_eval(highest_value_index_ga["Best Solution"]), "GA")