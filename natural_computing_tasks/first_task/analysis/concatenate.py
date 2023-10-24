import pandas as pd

df = pd.concat([pd.read_csv(f'results/results{i + 1}.csv') for i in range(3)])
df = df.groupby('experiment_type').agg(
    {'best_rmse': ['mean', 'std'], 'r2_score': ['mean', 'std']}
)
print(df)
