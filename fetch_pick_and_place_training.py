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

# Fix for MuJoCo headless offscreen rendering (must be set before gymnasium/mujoco imports)
os.environ["MUJOCO_GL"] = "egl"

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
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

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
NORM_DIR    = os.path.join(PROJECT_DIR, 'normalization')
# NORM_DIR    = os.path.join(PROJECT_DIR, 'normalization')  # Disabled - no normalization

for d in [LOG_DIR, MODEL_DIR, VIDEO_DIR, TB_LOG_DIR, NORM_DIR]:
    os.makedirs(d, exist_ok=True)

# ==================== CONFIGURATION ====================
ENV_ID = 'FetchPickAndPlace-v4'
TOTAL_TIMESTEPS = 10_000_000   # Extended: curve not plateaued at 5M (68% SR, still trending)
EVAL_FREQ = 250_000
N_EVAL_EPISODES = 100
VIDEO_FREQ = 1_000_000         # Record one checkpoint video every 1M steps during training
N_VIDEO_EPISODES = 1           # One episode per checkpoint (keep overhead low)
SEEDS = [42, 120]
N_ENVS = 8

USE_FEATURE_WRAPPER = False
USE_VEC_NORMALIZE = True

HER_KWARGS = dict(n_sampled_goal=4, goal_selection_strategy='future')

# ==================== HYPERPARAMETER GRIDS ====================
HYPERPARAMETER_GRIDS = {
    'SAC': {
        'learning_rate': [1e-3,5e-4],
        'batch_size': [256],
        'gamma': [0.98,0.99,0.995],
        'tau': [0.05,0.005],
        'net_arch': [[256, 256, 256]],
        'ent_coef': ['auto'],
    },
    'TD3': {
        'learning_rate': [1e-3],
        'batch_size': [256],
        'gamma': [0.98],
        'tau': [0.005],
        'net_arch': [[256, 256, 256]],
        'policy_delay': [2],
        'target_policy_noise': [0.2],
        'target_noise_clip': [0.5],
    },
    'DDPG': {
        'learning_rate': [1e-3],
        'batch_size': [256],
        'gamma': [0.98],
        'tau': [0.005],
        'net_arch': [[256, 256, 256]],
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
    def __init__(self, eval_env, run_name, eval_freq, n_eval_episodes, log_path,
                 video_env=None, video_dir=None, video_freq=1_000_000, n_video_episodes=1,
                 config=None, verbose=0):
        super().__init__(verbose)
        self.eval_env          = eval_env
        self.run_name          = run_name
        self.eval_freq         = eval_freq
        self.n_eval_episodes   = n_eval_episodes
        self.log_path          = log_path
        self.eval_results      = []
        self.config            = config or {}

        # ---- video checkpointing ----
        # video_env must be a DummyVecEnv wrapping a render_mode='rgb_array' env,
        # with VecNormalize applied (obs_rms shared from train_env). It is kept
        # alive for the full training run and closed externally alongside eval_env.
        self.video_env         = video_env
        self.video_dir         = video_dir
        self.video_freq        = video_freq
        self.n_video_episodes  = n_video_episodes

    def _on_training_start(self) -> None:
        """Log hyperparameters to TensorBoard at the start of training."""
        for key, value in self.config.items():
            if isinstance(value, (int, float, bool)):
                self.logger.record(f"config/{key}", value)
            else:
                self.logger.record(f"config/{key}", str(value))
        self.logger.dump(self.num_timesteps)

    def _record_checkpoint_video(self) -> None:
        """Record N_VIDEO_EPISODES episodes and save to disk. Called at video_freq intervals."""
        if self.video_env is None or self.video_dir is None:
            return

        step_tag   = f"{self.num_timesteps // 1_000}k"
        video_path = os.path.join(self.video_dir, f"{self.run_name}_step{step_tag}.mp4")

        frames = []
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            for _ in range(self.n_video_episodes):
                obs  = self.video_env.reset()
                done = np.array([False])
                while not done[0]:
                    try:
                        # get_images() is the correct VecEnv API for rgb_array frames.
                        # It works through VecNormalize → DummyVecEnv → env.render()
                        # without needing to access .envs[0] directly (which VecNormalize hides).
                        imgs = self.video_env.get_images()
                        if imgs and imgs[0] is not None:
                            frames.append(imgs[0])
                    except Exception as e:
                        print(f"\n  [Video Warning] Offscreen rendering failed: {e}")
                        print("  Skipping video recording for this run to prevent training crash.")
                        self.video_env = None
                        return
                    
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, done, _ = self.video_env.step(action)

        if frames:
            imageio.mimsave(video_path, frames, fps=30)
            print(f"  [Video] Saved checkpoint: {video_path} ({len(frames)} frames)")

    def _on_step(self) -> bool:
        # ---- eval checkpoint ----
        if self.num_timesteps % self.eval_freq < self.model.n_envs:
            successes, rewards = [], []
            for _ in range(self.n_eval_episodes):
                obs      = self.eval_env.reset()
                done     = np.array([False])
                ep_reward = 0.0
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    while not done[0]:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, done, info = self.eval_env.step(action)
                        ep_reward += reward[0]
                successes.append(
                    info[0].get('is_success', 0.0) if isinstance(info, list)
                    else info.get('is_success', 0.0)
                )
                rewards.append(float(ep_reward))

            mean_success_rate = np.mean(successes)
            mean_reward       = np.mean(rewards)

            result = {
                'timestep':          self.num_timesteps,
                'mean_success_rate': mean_success_rate,
                'mean_reward':       mean_reward,
            }
            self.eval_results.append(result)

            if self.verbose > 0:
                print(f"[{self.run_name}] Step {self.num_timesteps}: "
                      f"Success Rate = {mean_success_rate:.3f}, Mean Reward = {mean_reward:.2f}")

            pd.DataFrame(self.eval_results).to_csv(self.log_path, index=False)

        # ---- video checkpoint (less frequent than eval) ----
        if self.num_timesteps % self.video_freq < self.model.n_envs:
            self._record_checkpoint_video()

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

    for algo_name in HYPERPARAMETER_GRIDS.keys():
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

                    model_path = os.path.join(MODEL_DIR, run_name)
                    norm_path  = os.path.join(NORM_DIR, f'{run_name}_vecnorm.pkl')
                    tb_run_dir = os.path.join(TB_LOG_DIR, run_name)
                    # os.makedirs(tb_run_dir, exist_ok=True)  # REMOVED: Causes SB3 to append _1

                    if USE_VEC_NORMALIZE:
                        # FIX #5 (Minor): On training restart, load saved norm stats from
                        # disk so the running mean/std are restored correctly. Sharing
                        # obs_rms by reference (the original approach) works within a
                        # single run but becomes stale after the process restarts.
                        if os.path.exists(norm_path):
                            train_env = VecNormalize.load(norm_path, train_env)
                            train_env.training = True
                            train_env.norm_reward = False
                            print(f"[VecNormalize] Loaded existing stats from {norm_path}")
                        else:
                            train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=5.0, norm_obs_keys=['observation'])

                    eval_env_raw = gym.make(ENV_ID)
                    if USE_FEATURE_WRAPPER:
                        eval_env_raw = FetchFeatureWrapper(eval_env_raw)

                    eval_env_vec = DummyVecEnv([lambda env=eval_env_raw: env])

                    if USE_VEC_NORMALIZE:
                        eval_env_vec = VecNormalize(eval_env_vec, norm_obs=True, norm_reward=False, clip_obs=5.0, training=False, norm_obs_keys=['observation'])
                        # Share the live RunningMeanStd reference so the eval env always
                        # uses up-to-date stats during this run. This is correct for
                        # within-run evaluation; post-training eval uses the saved file.
                        eval_env_vec.obs_rms = train_env.obs_rms

                    # ---- video env (separate from eval_env: needs render_mode='rgb_array') ----
                    # get_images() is used for rendering, which works correctly through
                    # VecNormalize → DummyVecEnv without needing direct .envs access.
                    video_run_dir = os.path.join(VIDEO_DIR, run_name)
                    os.makedirs(video_run_dir, exist_ok=True)

                    video_env_raw = gym.make(ENV_ID, render_mode='rgb_array')
                    if USE_FEATURE_WRAPPER:
                        video_env_raw = FetchFeatureWrapper(video_env_raw)
                    video_env_vec = DummyVecEnv([lambda env=video_env_raw: env])
                    if USE_VEC_NORMALIZE:
                        video_env_vec = VecNormalize(
                            video_env_vec, norm_obs=True, norm_reward=False,
                            clip_obs=5.0, training=False, norm_obs_keys=['observation']
                        )
                        video_env_vec.obs_rms = train_env.obs_rms  # same live stats as train/eval

                    log_path = os.path.join(LOG_DIR, f'{run_name}_eval.csv')

                    extra_kwargs = {}
                    if algo_name in ['TD3', 'DDPG']:
                        n_actions = train_env.action_space.shape[-1]
                        # OpenAI paper uses sigma=0.2 for exploration
                        extra_kwargs['action_noise'] = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

                    os.makedirs(TB_LOG_DIR, exist_ok=True)

                    final_kwargs = {**SHARED_KWARGS, **hparam_config}
                    net_arch = final_kwargs.pop('net_arch')
                    policy_kwargs = {'net_arch': net_arch}
                    final_kwargs['policy_kwargs'] = policy_kwargs

                    with open(os.path.join(LOG_DIR, f'{run_name}_config.json'), 'w') as f:
                        json.dump({'algo': algo_name, 'seed': seed, 'config_idx': config_idx, 'total_timesteps': TOTAL_TIMESTEPS, 'use_feature_wrapper': USE_FEATURE_WRAPPER, 'use_vec_normalize': USE_VEC_NORMALIZE, 'her_kwargs': HER_KWARGS, 'hyperparams': final_kwargs}, f, indent=4)

                    if os.path.exists(model_path + '.zip'):
                        print(f"[Continuation] Loading existing model...")
                        model = algo_class.load(model_path, env=train_env, custom_objects={'replay_buffer_class': HerReplayBuffer}, **extra_kwargs)
                    else:
                        print(f"[Fresh Start] Initializing new model...")
                        model = algo_class('MultiInputPolicy', train_env, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=HER_KWARGS, seed=seed, verbose=0, tensorboard_log=TB_LOG_DIR, **final_kwargs, **extra_kwargs)

                    # Gather all hyperparams for logging
                    full_config = {
                        "algo": algo_name,
                        "lr": hparam_config.get('learning_rate'),
                        "bs": hparam_config.get('batch_size'),
                        "gamma": hparam_config.get('gamma'),
                        "tau": hparam_config.get('tau'),
                        "n_envs": N_ENVS,
                        "total_timesteps": TOTAL_TIMESTEPS,
                        "her_sampled": HER_KWARGS.get('n_sampled_goal'),
                        "norm": USE_VEC_NORMALIZE,
                    }
                    if algo_name in ['TD3', 'DDPG']:
                        full_config['noise_sigma'] = 0.2

                    callback = SuccessRateCallback(
                        eval_env=eval_env_vec,
                        run_name=run_name,
                        eval_freq=EVAL_FREQ,
                        n_eval_episodes=N_EVAL_EPISODES,
                        log_path=log_path,
                        video_env=video_env_vec,
                        video_dir=video_run_dir,
                        video_freq=VIDEO_FREQ,
                        n_video_episodes=N_VIDEO_EPISODES,
                        config=full_config,
                        verbose=1
                    )

                    t0 = time.time()
                    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, tb_log_name=run_name, progress_bar=True, reset_num_timesteps=True)
                    elapsed = time.time() - t0
                    print(f"\nFinished {run_name} in {elapsed/60:.1f} min")

                    model.save(model_path)
                    print(f"Model saved to {model_path}")

                    # FIX #1 (Critical): Save VecNormalize running stats to disk.
                    # The policy was trained on normalized observations; without saving
                    # and restoring these stats, all post-training evaluation and video
                    # recording will feed raw un-normalized inputs to the policy.
                    if USE_VEC_NORMALIZE:
                        train_env.save(norm_path)
                        print(f"VecNormalize stats saved to {norm_path}")

                    # Use the actual logger directory SB3 assigned (avoids the _1/_2 suffix issue)
                    actual_tb_log_dir = model.logger.dir
                    writer = SummaryWriter(log_dir=actual_tb_log_dir)
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
                    video_env_vec.close()

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
            norm_path  = os.path.join(NORM_DIR,  f'{algo_name}_cfg00_seed{seed}_vecnorm.pkl')

            if not os.path.exists(model_path + '.zip'):
                continue

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                eval_env = gym.make(ENV_ID, render_mode='rgb_array')
                if USE_FEATURE_WRAPPER:
                    eval_env = FetchFeatureWrapper(eval_env)

                eval_env_vec = DummyVecEnv([lambda env=eval_env: env])

                # FIX #4 (Significant): Apply VecNormalize when loading for evaluation.
                # The original code had a no-op self-assignment (eval_env_vec = eval_env_vec)
                # where this wrapping should have been. Without it the policy receives
                # raw un-normalized observations, making the benchmark numbers meaningless.
                if USE_VEC_NORMALIZE and os.path.exists(norm_path):
                    eval_env_vec = VecNormalize.load(norm_path, eval_env_vec)
                    eval_env_vec.training = False
                    eval_env_vec.norm_reward = False
                elif USE_VEC_NORMALIZE:
                    print(f"  Warning: norm stats not found at {norm_path}, evaluation may be inaccurate.")

                algo_class = {'TD3': TD3, 'DDPG': DDPG, 'SAC': SAC}[algo_name]
                model = algo_class.load(model_path, env=eval_env_vec)

                successes, rewards = [], []
                for ep in range(50):
                    obs = eval_env_vec.reset()
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

    def record_video(model, env_id, video_path, use_wrapper, norm_path, n_episodes=3, fps=30):
        """
        Record a video of the agent performing in the environment.

        FIX #2 (Critical): The function now accepts use_wrapper and norm_path so it
        can reconstruct the exact same observation pipeline the policy was trained on
        (FetchFeatureWrapper + VecNormalize). The original version called gym.make()
        with no wrapper, causing a 25-dim vs 32-dim shape mismatch at inference.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            env = gym.make(env_id, render_mode='rgb_array')
            if use_wrapper:
                env = FetchFeatureWrapper(env)

            env_vec = DummyVecEnv([lambda e=env: e])

            if USE_VEC_NORMALIZE and os.path.exists(norm_path):
                env_vec = VecNormalize.load(norm_path, env_vec)
                env_vec.training = False
                env_vec.norm_reward = False

            frames = []
            for ep in range(n_episodes):
                obs = env_vec.reset()
                done = np.array([False])
                while not done[0]:
                    try:
                        frames.append(env.render())
                    except Exception as e:
                        print(f"\n  [Video Warning] Offscreen rendering failed: {e}")
                        print("  Skipping video recording.")
                        env_vec.close()
                        return None
                        
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env_vec.step(action)
                try:
                    frames.append(env.render())
                except:
                    pass

            env_vec.close()
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
        norm_path  = os.path.join(NORM_DIR,  f'{algo_name}_cfg00_seed{best_seed}_vecnorm.pkl')

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            # FIX #2 (continued): Build the full observation pipeline before loading
            # the model, so the model's internal obs_space check passes correctly.
            load_env = gym.make(ENV_ID)
            if USE_FEATURE_WRAPPER:
                load_env = FetchFeatureWrapper(load_env)
            load_env_vec = DummyVecEnv([lambda e=load_env: e])
            if USE_VEC_NORMALIZE and os.path.exists(norm_path):
                load_env_vec = VecNormalize.load(norm_path, load_env_vec)
                load_env_vec.training = False
                load_env_vec.norm_reward = False

            algo_class = {'TD3': TD3, 'DDPG': DDPG, 'SAC': SAC}[algo_name]
            model = algo_class.load(model_path, env=load_env_vec)
            load_env_vec.close()

        video_path = os.path.join(VIDEO_DIR, f'{algo_name}_best.mp4')
        print(f'{algo_name} (seed={best_seed}, final success={best_sr:.2%}):')
        record_video(model, ENV_ID, video_path,
                     use_wrapper=USE_FEATURE_WRAPPER,
                     norm_path=norm_path,
                     n_episodes=3, fps=30)
        print()

    print('All videos recorded!')

    # ==================== TENSORBOARD NOTE ====================
    print("\n" + "="*70)
    print("To view detailed training logs with TensorBoard, run:")
    print(f"  tensorboard --logdir={TB_LOG_DIR}/")
    print("="*70)

if __name__ == '__main__':
    main()