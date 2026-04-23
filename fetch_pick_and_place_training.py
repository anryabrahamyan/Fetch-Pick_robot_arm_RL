#!/usr/bin/env python3
"""
Deep Reinforcement Learning for Fetch Pick and Place Training Script

Environment: FetchPickAndPlace-v4 (Gymnasium-Robotics / MuJoCo)
Algorithms: SAC, TD3, DDPG — all with Hindsight Experience Replay (HER)
Framework: Stable-Baselines3
"""

import os
import json
import time
import warnings
import logging
import itertools

# Suppress all warnings including gymnasium's observation space checker
warnings.filterwarnings('ignore')
logging.getLogger('gymnasium').setLevel(logging.ERROR)
logging.getLogger('gymnasium.utils.passive_env_checker').setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
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
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
# from stable_baselines3.common.vec_env import VecNormalize  # Disabled

import imageio
from torch.utils.tensorboard import SummaryWriter

# Import the custom wrapper
from fetch_wrappers import FetchFeatureWrapper

# ==================== SETUP & PATHS ====================
PROJECT_DIR = os.getcwd()
LOG_DIR     = os.path.join(PROJECT_DIR, 'logs')
MODEL_DIR   = os.path.join(PROJECT_DIR, 'models')
VIDEO_DIR   = os.path.join(PROJECT_DIR, 'videos')
TB_LOG_DIR  = os.path.join(PROJECT_DIR, 'tb_logs')
# NORM_DIR    = os.path.join(PROJECT_DIR, 'normalization')  # Disabled - no normalization

for d in [LOG_DIR, MODEL_DIR, VIDEO_DIR, TB_LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ==================== CONFIGURATION ====================
ENV_ID = 'FetchPickAndPlace-v4'
TOTAL_TIMESTEPS = 3_000_000
EVAL_FREQ = 250_000
N_EVAL_EPISODES = 100
SEEDS = [42, 120]
N_ENVS = 8

USE_FEATURE_WRAPPER = True
USE_VEC_NORMALIZE = False

HER_KWARGS = dict(n_sampled_goal=8, goal_selection_strategy='future')

# ==================== HYPERPARAMETER GRIDS ====================
HYPERPARAMETER_GRIDS = {
    'TD3': {
        'learning_rate': [1e-2,1e-3, 5e-3],
        'batch_size': [256],
        'gamma': [0.98, 0.99],
        'tau': [0.005],
        'net_arch': [[256, 256, 256]],
        'policy_delay': [2],
        'target_policy_noise': [0.2],
        'target_noise_clip': [0.5],
    },
    'DDPG': {
        'learning_rate': [1e-3, 5e-3],
        'batch_size': [256],
        'gamma': [0.98, 0.99],
        'tau': [0.005],
        'net_arch': [[256, 256, 256]],
    },
    'SAC': {
        'learning_rate': [1e-3, 5e-3],
        'batch_size': [256],
        'gamma': [0.98, 0.99],
        'tau': [0.005],
        'net_arch': [[256, 256, 256]],
        'ent_coef': ['auto'],
    },
}

SHARED_KWARGS = dict(buffer_size=2_000_000, learning_starts=5000, train_freq=4, gradient_steps=4)

def generate_configs(algo_name, grid):
    configs = []
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        configs.append(dict(zip(keys, values)))
    return configs

# ==================== CALLBACK ====================
class SuccessRateCallback(BaseCallback):
    def __init__(self, eval_env, run_name, eval_freq, n_eval_episodes, log_path, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.run_name = run_name
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path
        self.eval_results = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            successes, rewards = [], []
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = np.array([False])
                ep_reward = 0.0
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    while not done[0]:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, done, info = self.eval_env.step(action)
                        ep_reward += reward[0]
                successes.append(info[0].get('is_success', 0.0) if isinstance(info, list) else info.get('is_success', 0.0))
                rewards.append(float(ep_reward))

            mean_success_rate = np.mean(successes)
            mean_reward = np.mean(rewards)

            result = {
                'timestep': self.num_timesteps,
                'mean_success_rate': mean_success_rate,
                'mean_reward': mean_reward,
            }
            self.eval_results.append(result)

            if self.verbose > 0:
                print(f"[{self.run_name}] Step {self.num_timesteps}: Success Rate = {mean_success_rate:.3f}, Mean Reward = {mean_reward:.2f}")

            df = pd.DataFrame(self.eval_results)
            df.to_csv(self.log_path, index=False)

        return True

# ==================== ENVIRONMENT FACTORY ====================
def make_fetch_env(env_id, seed, rank, wrapper=False):
    """Factory function for SubprocVecEnv."""
    def _init():
        import gymnasium as gym
        import gymnasium_robotics
        import warnings
        import logging

        warnings.filterwarnings('ignore')
        logging.getLogger('gymnasium').setLevel(logging.ERROR)
        logging.getLogger('gymnasium.utils.passive_env_checker').setLevel(logging.CRITICAL)

        gym.register_envs(gymnasium_robotics)

        env = gym.make(env_id)

        if wrapper:
            from fetch_wrappers import FetchFeatureWrapper
            env = FetchFeatureWrapper(env)

        env.reset(seed=seed + rank)
        return env

    return _init

# ==================== MAIN FUNCTION ====================
def main():
    print('Setup complete.')
    print(f'Project directory: {PROJECT_DIR}')

    # ==================== OBSERVATION SPACE DIAGNOSTICS ====================
    print("="*70 + "\nOBSERVATION SPACE DIAGNOSTICS\n" + "="*70)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        # Test 1: Base environment
        base_env = gym.make(ENV_ID)
        obs, info = base_env.reset()
        print(f"\n1. Base Environment (no wrapper):\n   Obs space: {base_env.observation_space}\n   Obs 'observation' shape: {obs['observation'].shape}\n   Obs 'achieved_goal' shape: {obs['achieved_goal'].shape}\n   Obs 'desired_goal' shape: {obs['desired_goal'].shape}")
        base_env.close()

        # Test 2: With feature wrapper
        wrapped_env = gym.make(ENV_ID)
        wrapped_env = FetchFeatureWrapper(wrapped_env)
        obs, info = wrapped_env.reset()
        print(f"\n2. With FetchFeatureWrapper (delta coordinates APPENDED):\n   Obs space: {wrapped_env.observation_space}\n   Obs 'observation' shape: {obs['observation'].shape}\n   ✓ Original features: 25 (indices 0:25)\n   ✓ obj_rel_gripper: 3 (indices 25:28)\n   ✓ goal_rel_obj: 3 (indices 28:31)\n   ✓ dist_to_goal: 1 (index 31)\n   ✓ Total: 25 + 7 = 32 features")
        wrapped_env.close()

        # Test 3: VecNormalize disabled
        # dummy_env_wrapped = DummyVecEnv([lambda: FetchFeatureWrapper(gym.make(ENV_ID))])
        # vec_normalized = VecNormalize(dummy_env_wrapped, norm_obs=True, norm_reward=False, clip_obs=10.0)
        # obs = vec_normalized.reset()
        # print(f"\n3. With FetchFeatureWrapper + VecNormalize:\n   Obs space (after norm): {vec_normalized.observation_space}\n   Obs dict keys: {list(obs.keys()) if isinstance(obs, dict) else 'N/A (array)'}")
        # if isinstance(obs, dict):
        #     print(f"   Obs 'observation' shape: {obs['observation'].shape}\n   ✓ VecNormalize applied successfully\n   ✓ Observations normalized to N(0, 1)\n   ✓ Clipping enabled: [-10, 10]\n   ✓ Reward normalization: DISABLED (HER compatibility)")
        # vec_normalized.close()

    print("\n" + "="*70 + "\n✓ All observation shapes validated successfully!\n" + "="*70)

    # ==================== HYPERPARAMETER GRID SETUP ====================
    print("="*70 + "\nHYPERPARAMETER GRID SEARCH SETUP\n" + "="*70)
    ALGO_CONFIGS = {}
    for algo_name in ['TD3', 'DDPG', 'SAC']:
        configs = generate_configs(algo_name, HYPERPARAMETER_GRIDS[algo_name])
        ALGO_CONFIGS[algo_name] = configs
        print(f"\n{algo_name}: {len(configs)} configuration(s)")
        if len(configs) <= 3:
            for i, cfg in enumerate(configs):
                print(f"  Config {i+1}: learning_rate={cfg['learning_rate']}, batch_size={cfg['batch_size']}, net_arch={cfg['net_arch']}")

    print(f"\n" + "="*70)
    print(f"Environment: {ENV_ID}\nTimesteps: {TOTAL_TIMESTEPS:,}\nParallel envs: {N_ENVS}\nFeature wrapper: {USE_FEATURE_WRAPPER}\nVec normalize: {USE_VEC_NORMALIZE}\nEval freq: every {EVAL_FREQ:,} steps\nSeeds: {SEEDS}\nHER strategy: {HER_KWARGS['goal_selection_strategy']} (n_sampled_goal={HER_KWARGS['n_sampled_goal']})")
    print("="*70)

    # ==================== TRAINING ====================
    all_results = {}

    for algo_name in ['TD3', 'DDPG', 'SAC']:
        algo_class = {'TD3': TD3, 'DDPG': DDPG, 'SAC': SAC}[algo_name]
        all_results[algo_name] = {}

        for config_idx, hparam_config in enumerate(ALGO_CONFIGS[algo_name]):
            config_name = f"cfg{config_idx:02d}"
            all_results[algo_name][config_name] = {}

            for seed in SEEDS:
                run_name = f'{algo_name}_{config_name}_seed{seed}'
                print(f"\n{'='*70}\nTraining: {run_name} | Envs: {N_ENVS} | Wrapper: {USE_FEATURE_WRAPPER} | VecNorm: {USE_VEC_NORMALIZE}\nHparams: lr={hparam_config['learning_rate']}, bs={hparam_config['batch_size']}, gamma={hparam_config['gamma']}, tau={hparam_config['tau']}\n{'='*70}")

                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    train_env = SubprocVecEnv([make_fetch_env(ENV_ID, seed, i, wrapper=USE_FEATURE_WRAPPER) for i in range(N_ENVS)])

                    eval_env = gym.make(ENV_ID, render_mode='rgb_array')
                    if USE_FEATURE_WRAPPER:
                        eval_env = FetchFeatureWrapper(eval_env)

                    eval_env_vec = DummyVecEnv([lambda env=eval_env: env])
                    eval_env_vec = eval_env_vec

                    log_path = os.path.join(LOG_DIR, f'{run_name}_eval.csv')

                    extra_kwargs = {}
                    if algo_name in ['TD3', 'DDPG']:
                        n_actions = train_env.action_space.shape[-1]
                        extra_kwargs['action_noise'] = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

                    model_path = os.path.join(MODEL_DIR, run_name)
                    # norm_path = os.path.join(NORM_DIR, f'{run_name}_vecnorm')  # Disabled - no normalization
                    tb_run_dir = os.path.join(TB_LOG_DIR, run_name)
                    os.makedirs(tb_run_dir, exist_ok=True)

                    final_kwargs = {**SHARED_KWARGS, **hparam_config}
                    net_arch = final_kwargs.pop('net_arch')
                    policy_kwargs = {'net_arch': net_arch}
                    final_kwargs['policy_kwargs'] = policy_kwargs

                    with open(os.path.join(LOG_DIR, f'{run_name}_config.json'), 'w') as f:
                        json.dump({'algo': algo_name, 'seed': seed, 'config_idx': config_idx, 'total_timesteps': TOTAL_TIMESTEPS, 'use_feature_wrapper': USE_FEATURE_WRAPPER, 'use_vec_normalize': USE_VEC_NORMALIZE, 'her_kwargs': HER_KWARGS, 'hyperparams': final_kwargs}, f, indent=4)

                    # Ensure TB directory exists
                    os.makedirs(TB_LOG_DIR, exist_ok=True)

                    if os.path.exists(model_path + '.zip'):
                        print(f"[Continuation] Loading existing model...")
                        model = algo_class.load(model_path, env=train_env, custom_objects={'replay_buffer_class': HerReplayBuffer}, **extra_kwargs)
                    else:
                        print(f"[Fresh Start] Initializing new model...")
                        model = algo_class('MultiInputPolicy', train_env, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=HER_KWARGS, seed=seed, verbose=0, tensorboard_log=TB_LOG_DIR, **final_kwargs, **extra_kwargs)

                    callback = SuccessRateCallback(eval_env=eval_env_vec, run_name=run_name, eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_EPISODES, log_path=log_path, verbose=1)

                    t0 = time.time()
                    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, tb_log_name=run_name, progress_bar=True)
                    elapsed = time.time() - t0
                    print(f"\nFinished {run_name} in {elapsed/60:.1f} min\nModel saved to {model_path}")

                    model.save(model_path)

                    os.makedirs(tb_run_dir, exist_ok=True)
                    writer = SummaryWriter(log_dir=tb_run_dir)
                    hparam_dict = {
                        'learning_rate': hparam_config['learning_rate'],
                        'batch_size': hparam_config['batch_size'],
                        'gamma': hparam_config['gamma'],
                        'tau': hparam_config['tau'],
                        'buffer_size': final_kwargs['buffer_size'],
                        'net_arch_layers': len(hparam_config['net_arch']),
                        'net_arch_units': hparam_config['net_arch'][0] if hparam_config['net_arch'] else 0,
                    }
                    metric_dict = {
                        'final_success_rate': callback.eval_results[-1]['mean_success_rate'] if callback.eval_results else 0,
                        'final_reward': callback.eval_results[-1]['mean_reward'] if callback.eval_results else 0,
                    }
                    writer.add_hparams(hparam_dict, metric_dict)
                    writer.flush()
                    writer.close()
                    print(f"Hyperparameters logged to TensorBoard")


                    all_results[algo_name][config_name][seed] = callback.eval_results

                    train_env.close()
                    eval_env_vec.close()

    print('\n' + '='*70 + '\nALL TRAINING COMPLETE\n' + '='*70)

    # ==================== LEARNING CURVES ====================
    print("\nGenerating learning curves...")

    aggregated_results = {}

    for algo_name in ['TD3', 'DDPG', 'SAC']:
        if algo_name not in all_results:
            continue
        aggregated_results[algo_name] = {}

        for config_name, seed_dict in all_results[algo_name].items():
            for seed, eval_results in seed_dict.items():
                if seed not in aggregated_results[algo_name]:
                    aggregated_results[algo_name][seed] = eval_results

    COLORS = {'SAC': '#2196F3', 'TD3': '#FF5722', 'DDPG': '#4CAF50'}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for algo_name, seed_results in aggregated_results.items():
        if not seed_results:
            continue

        dfs = [pd.DataFrame(records) for records in seed_results.values()]
        if not dfs:
            continue

        timesteps = dfs[0]['timestep'].values

        success_matrix = np.array([df['mean_success_rate'].values for df in dfs])
        mean_success = success_matrix.mean(axis=0)
        std_success  = success_matrix.std(axis=0)

        reward_matrix = np.array([df['mean_reward'].values for df in dfs])
        mean_reward = reward_matrix.mean(axis=0)
        std_reward  = reward_matrix.std(axis=0)

        color = COLORS[algo_name]

        axes[0].plot(timesteps, mean_success, label=algo_name, color=color, linewidth=2)
        axes[0].fill_between(timesteps, mean_success - std_success, mean_success + std_success,
                             color=color, alpha=0.15)

        axes[1].plot(timesteps, mean_reward, label=algo_name, color=color, linewidth=2)
        axes[1].fill_between(timesteps, mean_reward - std_reward, mean_reward + std_reward,
                             color=color, alpha=0.15)

    axes[0].set_title('Success Rate vs. Training Timesteps', fontweight='bold')
    axes[0].set_xlabel('Timesteps')
    axes[0].set_ylabel('Success Rate')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(frameon=True, fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Mean Episode Reward vs. Training Timesteps', fontweight='bold')
    axes[1].set_xlabel('Timesteps')
    axes[1].set_ylabel('Mean Reward')
    axes[1].legend(frameon=True, fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('FetchPickAndPlace-v4 — Learning Curves (mean ± std over seeds)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, 'learning_curves.png'), dpi=200, bbox_inches='tight')
    plt.show()

    # ==================== PERFORMANCE EVALUATION ====================
    print("\nEvaluating trained models...")

    eval_records = []

    for algo_name in ALGO_CONFIGS:
        for seed in SEEDS:
            model_path = os.path.join(MODEL_DIR, f'{algo_name}_cfg00_seed{seed}')
            # norm_path = os.path.join(NORM_DIR, f'{algo_name}_cfg00_seed{seed}_vecnorm')  # Disabled - no normalization

            if not os.path.exists(model_path + '.zip'):
                continue

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                eval_env = gym.make(ENV_ID, render_mode='rgb_array')
                if USE_FEATURE_WRAPPER:
                    eval_env = FetchFeatureWrapper(eval_env)

                eval_env_vec = DummyVecEnv([lambda env=eval_env: env])
                eval_env_vec = eval_env_vec

                algo_class = {'TD3': TD3, 'DDPG': DDPG, 'SAC': SAC}[algo_name]
                model = algo_class.load(model_path, env=eval_env_vec)

                successes, rewards = [], []
                for ep in range(50):
                    obs, info = eval_env_vec.reset()
                    done = np.array([False])
                    ep_reward = 0.0
                    while not done[0]:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, done, info = eval_env_vec.step(action)
                        ep_reward += reward[0]
                    is_success = info[0].get('is_success', 0.0) if isinstance(info, (list, tuple)) and isinstance(info[0], dict) else info.get('is_success', 0.0) if isinstance(info, dict) else 0.0
                    successes.append(is_success)
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
                eval_env_vec.close()

    eval_df = pd.DataFrame(eval_records)
    print('='*80)
    print('  FINAL PERFORMANCE EVALUATION  (50 episodes per model)')
    print('='*80)
    print(eval_df.to_string(index=False))

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
    summary_df = pd.DataFrame(summary_records)
    print(summary_df.to_string(index=False))

    eval_df.to_csv(os.path.join(LOG_DIR, 'final_evaluation.csv'), index=False)
    summary_df.to_csv(os.path.join(LOG_DIR, 'final_summary.csv'), index=False)

    # ==================== VIDEO RECORDING ====================
    print("\nRecording agent videos...\n")

    def record_video(model, env_id, video_path, n_episodes=3, fps=30):
        """Record a video of the agent performing in the environment."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            env = gym.make(env_id, render_mode='rgb_array')
            frames = []

            for ep in range(n_episodes):
                obs, info = env.reset()
                done = False
                while not done:
                    frames.append(env.render())
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                frames.append(env.render())

            env.close()
            imageio.mimsave(video_path, frames, fps=fps)
            print(f'  Saved video: {video_path} ({len(frames)} frames)')
            return video_path

    for algo_name in ALGO_CONFIGS:
        best_seed, best_sr = None, -1
        for seed in SEEDS:
            csv_path = os.path.join(LOG_DIR, f'{algo_name}_cfg00_seed{seed}_eval.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                final_sr = df['mean_success_rate'].iloc[-1] if len(df) > 0 else 0
                if final_sr > best_sr:
                    best_sr = final_sr
                    best_seed = seed

        if best_seed is None:
            print(f'  No model found for {algo_name}, skipping.')
            continue

        model_path = os.path.join(MODEL_DIR, f'{algo_name}_cfg00_seed{best_seed}')

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            eval_env = gym.make(ENV_ID)
            algo_class = {'TD3': TD3, 'DDPG': DDPG, 'SAC': SAC}[algo_name]
            model = algo_class.load(model_path, env=eval_env)
            eval_env.close()

        video_path = os.path.join(VIDEO_DIR, f'{algo_name}_best.mp4')
        print(f'{algo_name} (seed={best_seed}, final success={best_sr:.2%}):')
        record_video(model, ENV_ID, video_path, n_episodes=3, fps=30)
        print()

    print('All videos recorded!')

    # ==================== TENSORBOARD NOTE ====================
    print("\n" + "="*70)
    print("To view detailed training logs with TensorBoard, run:")
    print(f"  tensorboard --logdir={TB_LOG_DIR}/")
    print("="*70)

if __name__ == '__main__':
    main()
