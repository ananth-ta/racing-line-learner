import pandas as pd
import numpy as np

df = pd.read_csv('maps/vegas_optimal_line.csv')
dist = np.sqrt(df['x']**2 + df['y']**2)
spawn_idx = dist.idxmin()

next_idx = (spawn_idx + 1) % len(df)
prev_idx = (spawn_idx - 1) % len(df)

x_curr = df.iloc[spawn_idx]['x']
x_next = df.iloc[next_idx]['x']

if x_next > x_curr:
    print(f"Forward is +X. Shifting array so index {spawn_idx} becomes 0.")
    new_df = pd.concat([df.iloc[spawn_idx:], df.iloc[:spawn_idx]]).reset_index(drop=True)
else:
    print(f"Backward is +X. Reversing list and shifting so index {spawn_idx} becomes 0.")
    rev_df = df.iloc[::-1].reset_index(drop=True)
    new_spawn_idx = len(df) - 1 - spawn_idx
    new_df = pd.concat([rev_df.iloc[new_spawn_idx:], rev_df.iloc[:new_spawn_idx]]).reset_index(drop=True)

new_df.to_csv('maps/vegas_optimal_line.csv', index=False)
