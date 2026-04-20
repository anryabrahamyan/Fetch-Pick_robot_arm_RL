import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import imageio
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.figsize': (12, 6),
    'figure.dpi': 120,
})

import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

from stable_baselines3 import SAC, TD3, DDPG, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv

# Paths
PROJECT_DIR = os.getcwd()
LOG_DIR     = os.path.join(PROJECT_DIR, 'logs')
MODEL_DIR   = os.path.join(PROJECT_DIR, 'models')
VIDEO_DIR   = os.path.join(PROJECT_DIR, 'videos')
TB_LOG_DIR  = os.path.join(PROJECT_DIR, 'tb_logs')

for d in [LOG_DIR, MODEL_DIR, VIDEO_DIR, TB_LOG_DIR]:
    os.makedirs(d, exist_ok=True)

print('Setup complete.')
print(f'Project directory: {PROJECT_DIR}')

# ==================== CONFIGURATION ====================
ENV_ID          = 'FetchPickAndPlace-v4'
TOTAL_TIMESTEPS = 2_000_000        # Increase for better convergence (1M+ recommended)
EVAL_FREQ       = 25_000          # Evaluate every N steps
N_EVAL_EPISODES = 100              # Episodes per evaluation
SEEDS           = [42]             # Random seeds for reproducibility
N_ENVS          = 8               # Number of parallel environments (multiprocessing)
network_layer_neurons = 256

# HER configuration (shared across all algorithms)
HER_KWARGS = dict(
    n_sampled_goal=8,
    goal_selection_strategy='future',
)

# Per-algorithm hyperparameters (tuned for Fetch environments)
ALGO_CONFIGS = {
    'SAC': {
        'class': SAC,
        'kwargs': dict(
            learning_rate=5e-4,
            buffer_size=1_000_000,
            policy_kwargs=dict(net_arch=[network_layer_neurons, network_layer_neurons]),
            batch_size=256,
            gamma=0.98,
            tau=0.005,
            learning_starts=1000,
            train_freq=4,
            gradient_steps=4
        ),
    },
    'TD3': {
        'class': TD3,
        'kwargs': dict(
            learning_rate=5e-4,
            buffer_size=1_000_000,
            policy_kwargs=dict(net_arch=[network_layer_neurons, network_layer_neurons, network_layer_neurons]),
            batch_size=256,
            gamma=0.98,
            tau=0.005,
            learning_starts=1000,
            train_freq=4,
            gradient_steps=4
        ),
    },
    'DDPG': {
        'class': DDPG,
        'kwargs': dict(
            learning_rate=5e-4,
            buffer_size=1_000_000,
            batch_size=256,
            policy_kwargs=dict(net_arch=[network_layer_neurons, network_layer_neurons, network_layer_neurons]),
            gamma=0.98,
            tau=0.005,
            learning_starts=1000,
            train_freq=4,
            gradient_steps=4
        ),
    },
}

# ==================== ENV FACTORY ====================
def make_fetch_env(env_id, seed, rank):
    """
    Factory function for SubprocVecEnv.
    Each subprocess must register gymnasium_robotics envs independently.
    """
    def _init():
        import gymnasium as gym
        import gymnasium_robotics
        gym.register_envs(gymnasium_robotics)
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env
    return _init

# ==================== CALLBACK ====================
class SuccessRateCallback(BaseCallback):
    """Logs success rate and episode reward at regular intervals."""
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=20, log_path=None, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path
        self.eval_results = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            successes, rewards_list = [], []
            for _ in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                terminated, truncated = False, False
                ep_reward = 0.0
                while not (terminated or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    ep_reward += reward
                successes.append(info.get('is_success', 0.0))
                rewards_list.append(ep_reward)

            mean_success = np.mean(successes)
            mean_reward  = np.mean(rewards_list)
            std_reward   = np.std(rewards_list)

            self.eval_results.append({
                'timestep': self.num_timesteps,
                'mean_success_rate': mean_success,
                'mean_reward': mean_reward,
                'std_reward': std_reward,
            })

            if self.verbose:
                print(f'  Step {self.num_timesteps:>8,} | '
                      f'Success: {mean_success:.2%} | '
                      f'Reward: {mean_reward:.1f} ± {std_reward:.1f}')

            if self.log_path:
                pd.DataFrame(self.eval_results).to_csv(self.log_path, index=False)

        return True

# ==================== MAIN TRAINING PIPELINE ====================
def main():
    print(f'Environment:      {ENV_ID}')
    print(f'Timesteps:        {TOTAL_TIMESTEPS:,}')
    print(f'Parallel envs:    {N_ENVS} (SubprocVecEnv)')
    print(f'Eval frequency:   every {EVAL_FREQ:,} steps')
    print(f'Seeds:            {SEEDS}')
    print(f'Algorithms:       {", ".join(ALGO_CONFIGS.keys())}')
    print(f'HER strategy:     {HER_KWARGS["goal_selection_strategy"]} (n_sampled_goal={HER_KWARGS["n_sampled_goal"]})')

    all_results = {}  # {algo_name: {seed: [eval_records]}}

    for algo_name, config in ALGO_CONFIGS.items():
        all_results[algo_name] = {}

        for seed in SEEDS:
            run_name = f'{algo_name}_seed{seed}'
            print(f'\n{"="*60}')
            print(f'  Training {run_name}  ({N_ENVS} parallel envs)')
            print(f'{"="*60}')

            # Create vectorized training env with SubprocVecEnv
            train_env = SubprocVecEnv(
                [make_fetch_env(ENV_ID, seed, i) for i in range(N_ENVS)]
            )

            # Single eval env
            eval_env = gym.make(ENV_ID)

            # Log path for this run
            log_path = os.path.join(LOG_DIR, f'{run_name}_eval.csv')

            # Action noise for TD3 and DDPG
            extra_kwargs = {}
            if algo_name in ['TD3', 'DDPG']:
                n_actions = train_env.action_space.shape[-1]
                extra_kwargs['action_noise'] = NormalActionNoise(
                    mean=np.zeros(n_actions),
                    sigma=0.1 * np.ones(n_actions)
                )

            # Build model
            model = config['class'](
                'MultiInputPolicy',
                train_env,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=HER_KWARGS,
                seed=seed,
                verbose=0,
                tensorboard_log=TB_LOG_DIR,
                **config['kwargs'],
                **extra_kwargs,
            )

            # Callback
            callback = SuccessRateCallback(
                eval_env=eval_env,
                eval_freq=EVAL_FREQ // N_ENVS,
                n_eval_episodes=N_EVAL_EPISODES,
                log_path=log_path,
                verbose=1,
            )

            # Train
            t0 = time.time()
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=callback,
                tb_log_name=run_name,
                progress_bar=True,
            )
            elapsed = time.time() - t0
            print(f'  Finished {run_name} in {elapsed/60:.1f} min')

            # Save model
            model_path = os.path.join(MODEL_DIR, run_name)
            model.save(model_path)
            print(f'  Model saved to {model_path}')

            # Store results
            all_results[algo_name][seed] = callback.eval_results

            train_env.close()
            eval_env.close()

    print('\n' + '='*60)
    print('  ALL TRAINING COMPLETE')
    print('='*60)

    # ==================== 4. LEARNING CURVES ====================
    print('\nGenerating Learning Curves...')
    plt.figure(figsize=(12, 6))
    for algo_name in ALGO_CONFIGS:
        algo_success = []
        algo_timesteps = None
        for seed in SEEDS:
            csv_path = os.path.join(LOG_DIR, f'{algo_name}_seed{seed}_eval.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                algo_success.append(df['mean_success_rate'].values)
                algo_timesteps = df['timestep'].values
        if algo_success and algo_timesteps is not None:
            mean_success = np.mean(algo_success, axis=0) * 100
            std_success = np.std(algo_success, axis=0) * 100
            plt.plot(algo_timesteps, mean_success, label=algo_name)
            plt.fill_between(algo_timesteps, 
                             mean_success - std_success, 
                             mean_success + std_success, alpha=0.2)
            
    plt.title('FetchPickAndPlace-v4 - Learning Curves (Success Rate)')
    plt.xlabel('Timesteps')
    plt.ylabel('Success Rate (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    curves_path = os.path.join(LOG_DIR, 'learning_curves.png')
    try:
        plt.savefig(curves_path)
        print(f'  Saved learning curves to {curves_path}')
    except Exception as e:
        print(f'  Could not save learning curves: {e}')

    # ==================== 5. PERFORMANCE EVALUATION ====================
    print('\n' + '='*60)
    print('  FINAL PERFORMANCE EVALUATION  (50 episodes per model)')
    print('='*60)
    eval_records = []
    for algo_name in ALGO_CONFIGS:
        for seed in SEEDS:
            model_path = os.path.join(MODEL_DIR, f'{algo_name}_seed{seed}')
            if not os.path.exists(model_path + '.zip'):
                continue
            
            eval_env = gym.make(ENV_ID)
            try:
                model = ALGO_CONFIGS[algo_name]['class'].load(model_path, env=eval_env)
            except Exception as e:
                print(f"Could not load {algo_name} seed {seed}: {e}")
                continue

            successes, rewards = [], []
            for ep in range(50):
                obs, info = eval_env.reset()
                done = False
                ep_reward = 0.0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    ep_reward += reward
                    done = terminated or truncated
                successes.append(info.get('is_success', 0.0))
                rewards.append(ep_reward)

            eval_records.append({
                'Algorithm': algo_name,
                'Seed': seed,
                'Success Rate (%)': f"{np.mean(successes)*100:.1f}",
                'Mean Reward': f"{np.mean(rewards):.2f}",
                'Std Reward': f"{np.std(rewards):.2f}",
                'Min Reward': f"{np.min(rewards):.2f}",
                'Max Reward': f"{np.max(rewards):.2f}",
            })
            eval_env.close()

    eval_df = pd.DataFrame(eval_records)
    print(eval_df.to_string(index=False))
    eval_df.to_csv(os.path.join(LOG_DIR, 'final_evaluation.csv'), index=False)

    print('\n--- Average Performance Across Seeds ---')
    summary_records = []
    for algo_name in ALGO_CONFIGS:
        algo_rows = [r for r in eval_records if r['Algorithm'] == algo_name]
        if algo_rows:
            avg_sr = np.mean([float(r['Success Rate (%)']) for r in algo_rows])
            avg_rw = np.mean([float(r['Mean Reward']) for r in algo_rows])
            summary_records.append({
                'Algorithm': algo_name,
                'Avg Success Rate (%)': f'{avg_sr:.1f}',
                'Avg Mean Reward': f'{avg_rw:.2f}',
            })
    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(os.path.join(LOG_DIR, 'final_summary.csv'), index=False)

    # ==================== 6. VIDEO RECORDING & BEST MODEL TRACKING ====================
    print('\nRecording agent videos and tracking best models...\n')
    best_models_info = {}
    global_best_sr = -1
    global_best_info = None

    for algo_name in ALGO_CONFIGS:
        best_seed, best_sr = None, -1
        for seed in SEEDS:
            csv_path = os.path.join(LOG_DIR, f'{algo_name}_seed{seed}_eval.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                final_sr = df['mean_success_rate'].iloc[-1] if len(df) > 0 else 0
                if final_sr >= best_sr:
                    best_sr = final_sr
                    best_seed = seed

        if best_seed is None:
            print(f'  No model found for {algo_name}, skipping.')
            continue

        model_name = f'{algo_name}_seed{best_seed}'
        model_path = os.path.join(MODEL_DIR, model_name)
        
        # Save algo params
        algo_params = ALGO_CONFIGS[algo_name]['kwargs'].copy()
        best_models_info[algo_name] = {
            'model_name': model_name,
            'success_rate': float(best_sr),
            'seed': best_seed,
            'hyperparameters': algo_params,
            'her_kwargs': HER_KWARGS
        }

        if best_sr > global_best_sr:
            global_best_sr = best_sr
            global_best_info = {
                'algorithm': algo_name,
                'model_name': model_name,
                'success_rate': float(best_sr),
            }

        if not os.path.exists(model_path + '.zip'):
            continue
            
        eval_env = gym.make(ENV_ID)
        try:
            model = ALGO_CONFIGS[algo_name]['class'].load(model_path, env=eval_env)
        except Exception:
            eval_env.close()
            continue
        eval_env.close()

        video_path = os.path.join(VIDEO_DIR, f'{algo_name}_best.mp4')
        print(f'{algo_name} (seed={best_seed}, final success={best_sr:.2%}):')
        
        env = gym.make(ENV_ID, render_mode='rgb_array')
        frames = []
        for ep in range(3):
            obs, info = env.reset()
            done = False
            while not done:
                frames.append(env.render())
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            frames.append(env.render())
        env.close()
        
        try:
            imageio.mimsave(video_path, frames, fps=30)
            print(f'  Saved video: {video_path} ({len(frames)} frames)\n')
        except Exception as e:
            print(f'  Could not save video due to error: {e}')

    best_models_info['GLOBAL_BEST'] = global_best_info
    info_path = os.path.join(LOG_DIR, 'best_models_info.json')
    try:
        with open(info_path, 'w') as f:
            json.dump(best_models_info, f, indent=4)
        print(f'Saved best models info to {info_path}')
    except Exception as e:
        print(f'Could not save best models info: {e}')

    print('All videos recorded!')

if __name__ == '__main__':
    main()
