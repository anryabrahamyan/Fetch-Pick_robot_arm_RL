import numpy as np
import gymnasium as gym

class FetchFeatureWrapper(gym.ObservationWrapper):
    """
    Custom wrapper for Fetch environments to enrich the observation space with delta coordinates.

    Appends the following features (does NOT replace):
    - obj_rel_gripper: 3D relative position (object - gripper)
    - goal_rel_obj: 3D relative position (goal - object)
    - dist_to_goal: 1D scalar Euclidean distance (goal - object)

    These are computed from the existing observation indices without removing any original features.
    """
    def __init__(self, env):
        super().__init__(env)
        # Original 'observation' shape is 25.
        # We APPEND (not replace):
        #   - 3 (obj_rel_gripper)
        #   - 3 (goal_rel_obj)
        #   - 1 (dist_to_goal)
        # Total new features: 7
        # Final observation shape: 25 + 7 = 32

        base_space = env.observation_space.spaces['observation']
        low = np.concatenate([
            base_space.low.astype(np.float32),
            np.full(7, -np.inf, dtype=np.float32)
        ])
        high = np.concatenate([
            base_space.high.astype(np.float32),
            np.full(7, np.inf, dtype=np.float32)
        ])

        # Update the specific 'observation' subspace
        self.observation_space.spaces['observation'] = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )

    def observation(self, obs):
        """
        Enhance observation by appending delta coordinates (not replacing).
        """
        # Extract features from dict observation, ensuring float32 from the start
        observation = np.asarray(obs['observation'], dtype=np.float32).copy()
        achieved_goal = np.asarray(obs['achieved_goal'], dtype=np.float32)
        desired_goal = np.asarray(obs['desired_goal'], dtype=np.float32)

        # Extract key positions from observation array
        # Index mapping (FetchPickAndPlace observation):
        gripper_pos = observation[0:3]       # Gripper position (x, y, z)
        object_pos = observation[10:13]      # Object position (x, y, z)

        # Compute delta coordinates (appended features)
        # 1. Relative position: object w.r.t. gripper
        obj_rel_gripper = (object_pos - gripper_pos).astype(np.float32)

        # 2. Relative position: goal w.r.t. object
        goal_rel_obj = (desired_goal - object_pos).astype(np.float32)

        # 3. Distance to goal (scalar)
        dist_to_goal = float(np.linalg.norm(desired_goal - achieved_goal))
        dist_to_goal = np.array([dist_to_goal], dtype=np.float32)

        # Append delta coordinates to original observation (do NOT replace)
        augmented_obs = np.concatenate([
            observation,      # 25 original features
            obj_rel_gripper,  # 3 appended: object-gripper delta
            goal_rel_obj,     # 3 appended: goal-object delta
            dist_to_goal      # 1 appended: distance to goal
        ], dtype=np.float32)

        # Return updated observation dict
        obs_copy = obs.copy()
        obs_copy['observation'] = augmented_obs
        obs_copy['achieved_goal'] = achieved_goal
        obs_copy['desired_goal'] = desired_goal
        return obs_copy
