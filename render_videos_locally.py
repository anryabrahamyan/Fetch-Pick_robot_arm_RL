import os
import warnings
import gymnasium as gym
import gymnasium_robotics
import imageio
import numpy as np
from stable_baselines3 import SAC, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from fetch_wrappers import FetchFeatureWrapper

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Configuration
ENV_ID = 'FetchPickAndPlace-v4'
MODEL_DIR = 'models'
NORM_DIR = 'normalization'
VIDEO_DIR = 'videos_local'
USE_FEATURE_WRAPPER = False
USE_VEC_NORMALIZE = True

os.makedirs(VIDEO_DIR, exist_ok=True)
gym.register_envs(gymnasium_robotics)

def record_video(model_path, norm_path, video_path, n_episodes=5, fps=50):
    print(f"Loading model from {model_path}...")
    
    # Setup environment for rendering
    env = gym.make(ENV_ID, render_mode='rgb_array')
    if USE_FEATURE_WRAPPER:
        env = FetchFeatureWrapper(env)
        
    env_vec = DummyVecEnv([lambda e=env: e])
    
    # Load normalization stats if they exist
    if USE_VEC_NORMALIZE and os.path.exists(norm_path):
        env_vec = VecNormalize.load(norm_path, env_vec)
        env_vec.training = False
        env_vec.norm_reward = False
        print(f"Loaded normalization stats from {norm_path}")
    else:
        print("Warning: Normalization stats not found. Policy may behave poorly.")

    # Determine algorithm from filename
    filename = os.path.basename(model_path)
    if 'SAC' in filename: algo_class = SAC
    elif 'TD3' in filename: algo_class = TD3
    elif 'DDPG' in filename: algo_class = DDPG
    else: algo_class = SAC # fallback

    # Load model
    model = algo_class.load(model_path, env=env_vec)
    
    frames = []
    for ep in range(n_episodes):
        obs = env_vec.reset()
        done = np.array([False])
        while not done[0]:
            frames.append(env.render())
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env_vec.step(action)
        frames.append(env.render())

    env_vec.close()
    
    if frames:
        imageio.mimsave(video_path, frames, fps=fps)
        print(f"Saved video to {video_path} ({len(frames)} frames)\n")

def main():
    print("Searching for trained models...")
    found_models = False
    
    if not os.path.exists(MODEL_DIR):
        print(f"Directory '{MODEL_DIR}' not found. Please download it from the server.")
        return
        
    for file in os.listdir(MODEL_DIR):
        if file.endswith('.zip'):
            found_models = True
            model_name = file.replace('.zip', '')
            model_path = os.path.join(MODEL_DIR, model_name)
            
            # Handle the '_best' suffix correctly for normalization files
            if model_name.endswith('_best'):
                base_name = model_name[:-5]
                norm_path = os.path.join(NORM_DIR, f"{base_name}_vecnorm_best.pkl")
            else:
                norm_path = os.path.join(NORM_DIR, f"{model_name}_vecnorm.pkl")
                
            video_path = os.path.join(VIDEO_DIR, f"{model_name}_generated.mp4")
            
            record_video(model_path, norm_path, video_path)
            
    if not found_models:
        print("No trained models found in the 'models' directory.")

if __name__ == '__main__':
    main()
