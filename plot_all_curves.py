import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.figsize': (14, 7),
    'figure.dpi': 120,
})

LOG_DIR = 'logs'

def main():
    # Find all eval CSVs
    csv_files = glob.glob(os.path.join(LOG_DIR, '*_eval.csv'))
    
    # Group by algorithm and experiment prefix
    grouped_results = {}
    
    for f in csv_files:
        basename = os.path.basename(f)
        
        if 'TD3' in basename:
            group_name = 'TD3'
        elif 'DDPG' in basename:
            group_name = 'DDPG'
        elif 'SAC' in basename:
            group_name = 'SAC'
        else:
            continue
        
        df = pd.read_csv(f)
        if group_name not in grouped_results:
            grouped_results[group_name] = []
        
        grouped_results[group_name].append(df)

    if not grouped_results:
        print("No evaluation CSVs found.")
        return

    fig, axes = plt.subplots(1, 2)
    
    # Generate colors automatically
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(grouped_results))]

    for idx, (group_name, dfs) in enumerate(sorted(grouped_results.items())):
        # Ensure all dataframes have the same timesteps by forward-filling shorter runs
        # (so early-stopped/collapsed runs don't cut off the entire average curve)
        longest_df = max(dfs, key=len)
        timesteps = longest_df['timestep'].values
        
        padded_dfs = []
        for df in dfs:
            df_indexed = df.set_index('timestep')
            # Reindex to the full timesteps and forward-fill missing values
            df_padded = df_indexed.reindex(timesteps).ffill().reset_index()
            padded_dfs.append(df_padded)
            
        success_matrix = np.array([df['mean_success_rate'].values for df in padded_dfs])
        mean_success = success_matrix.mean(axis=0)
        std_success  = success_matrix.std(axis=0)

        reward_matrix = np.array([df['mean_reward'].values for df in padded_dfs])
        mean_reward = reward_matrix.mean(axis=0)
        std_reward  = reward_matrix.std(axis=0)

        color = colors[idx]

        axes[0].plot(timesteps, mean_success, label=group_name, color=color, linewidth=2)
        axes[0].fill_between(timesteps, mean_success - std_success, mean_success + std_success,
                             color=color, alpha=0.15)

        axes[1].plot(timesteps, mean_reward, label=group_name, color=color, linewidth=2)
        axes[1].fill_between(timesteps, mean_reward - std_reward, mean_reward + std_reward,
                             color=color, alpha=0.15)

    axes[0].set_title('Success Rate vs. Training Timesteps', fontweight='bold')
    axes[0].set_xlabel('Timesteps')
    axes[0].set_ylabel('Success Rate')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(frameon=True, fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Mean Episode Reward vs. Training Timesteps', fontweight='bold')
    axes[1].set_xlabel('Timesteps')
    axes[1].set_ylabel('Mean Reward')
    axes[1].legend(frameon=True, fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('FetchPickAndPlace-v4 — Comprehensive Learning Curves',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(LOG_DIR, 'learning_curves_all.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved comprehensive learning curves to {out_path}")

if __name__ == '__main__':
    main()
