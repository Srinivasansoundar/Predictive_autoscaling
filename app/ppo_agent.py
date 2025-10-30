# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.vec_env import DummyVecEnv
# import logging
# from typing import Dict, Any, Tuple
# import json

# logger = logging.getLogger(__name__)


# class KubernetesScalingEnv(gym.Env):
#     """Custom Gym environment for Kubernetes pod autoscaling"""
    
#     metadata = {'render_modes': ['human']}
    
#     def __init__(self, 
#                  min_replicas: int = 1,
#                  max_replicas: int = 10,
#                  target_latency_ms: float = 1000.0,
#                  max_cpu_util: float = 80.0):
#         super().__init__()
        
#         self.min_replicas = min_replicas
#         self.max_replicas = max_replicas
#         self.target_latency_ms = target_latency_ms
#         self.max_cpu_util = max_cpu_util
        
#         # Define action space: discrete actions [-2, -1, 0, +1, +2] replicas
#         self.action_space = spaces.Discrete(5)
        
#         # Define observation space matching your state vector
#         self.observation_space = spaces.Dict({
#             'workload': spaces.Box(
#                 low=np.array([0, 0, 0, 0], dtype=np.float32),
#                 high=np.array([1000, 60000, 100, 1000], dtype=np.float32),
#                 dtype=np.float32
#             ),
#             'infra': spaces.Box(
#                 low=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
#                 high=np.array([20, 20, 100, 100, 10, 2, 2], dtype=np.float32),
#                 dtype=np.float32
#             ),
#             'time': spaces.Box(
#                 low=np.array([-1, -1, 0], dtype=np.float32),
#                 high=np.array([1, 1, 6], dtype=np.float32),
#                 dtype=np.float32
#             ),
#             'scaling': spaces.Box(
#                 low=np.array([-10, 0], dtype=np.float32),
#                 high=np.array([10, 100], dtype=np.float32),
#                 dtype=np.float32
#             ),
#             'trend': spaces.Box(
#                 low=np.array([-100, -100, -100], dtype=np.float32),
#                 high=np.array([100, 100, 100], dtype=np.float32),
#                 dtype=np.float32
#             )
#         })
        
#         self.current_state = None
#         self.last_replicas = min_replicas
#         self.episode_step = 0
#         self.max_episode_steps = 100
        
#     def _state_dict_to_obs(self, state_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
#         """Convert state dictionary to gym observation format"""
#         return {
#             'workload': np.array([
#                 state_dict['workload']['rps'],
#                 state_dict['workload']['p95_latency_ms'],
#                 state_dict['workload']['error_rate_pct'],
#                 state_dict['workload']['queue_length']
#             ], dtype=np.float32),
#             'infra': np.array([
#                 state_dict['infra']['pods_ready'],
#                 state_dict['infra']['hpa_desired_replicas'],
#                 state_dict['infra']['cpu_utilization_pct'],
#                 state_dict['infra']['mem_utilization_pct'],
#                 state_dict['infra']['node_count'],
#                 state_dict['infra']['pod_cpu_request_cores'],
#                 state_dict['infra']['pod_cpu_limit_cores']
#             ], dtype=np.float32),
#             'time': np.array([
#                 state_dict['time']['minute_of_day_sin'],
#                 state_dict['time']['minute_of_day_cos'],
#                 state_dict['time']['day_of_week']
#             ], dtype=np.float32),
#             'scaling': np.array([
#                 state_dict['scaling']['last_action_delta'],
#                 state_dict['scaling']['steps_since_action']
#             ], dtype=np.float32),
#             'trend': np.array([
#                 state_dict['trend']['cpu_slope'],
#                 state_dict['trend']['rps_slope'],
#                 state_dict['trend']['latency_slope']
#             ], dtype=np.float32)
#         }
    
#     def _action_to_delta(self, action: int) -> int:
#         """Convert discrete action to replica delta"""
#         action_map = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
#         return action_map[action]
    
#     def _calculate_reward(self, state_dict: Dict[str, Any], action_delta: int) -> float:
#         """Calculate reward based on system metrics and predictive trends"""
#         workload = state_dict['workload']
#         infra = state_dict['infra']
#         trend = state_dict.get('trend', {'cpu_slope': 0, 'rps_slope': 0, 'latency_slope': 0})
#         cpu_trend = trend.get('cpu_slope', 0)
#         rps_trend = trend.get('rps_slope', 0)
#         latency_trend = trend.get('latency_slope', 0)

#         # Latency penalty (normalized)
#         latency_penalty = (workload['p95_latency_ms'] / self.target_latency_ms) - 1.0
#         latency_penalty = max(0, latency_penalty) * 10  # Heavy penalty for exceeding target

#         # Error penalty
#         error_penalty = workload['error_rate_pct'] * 5.0

#         # Resource efficiency reward
#         cpu_util = infra['cpu_utilization_pct']
#         if cpu_util > self.max_cpu_util:
#             cpu_penalty = (cpu_util - self.max_cpu_util) / 10.0
#         else:
#             cpu_penalty = 0

#         # Resource waste penalty
#         if cpu_util < 30 and infra['pods_ready'] > self.min_replicas:
#             waste_penalty = (30 - cpu_util) / 20.0
#         else:
#             waste_penalty = 0

#         # Scaling smoothness penalty
#         scaling_penalty = abs(action_delta) * 0.5

#         # ----- TREND/ANTICIPATORY REWARD LOGIC -----
#         trend_bonus = 0
#         # If cpu or rps rising sharply, reward upscaling
#         if (cpu_trend > 5 or rps_trend > 10) and action_delta > 0:
#             trend_bonus += 2.0
#         # If cpu/rps dropping, reward downscaling
#         if (cpu_trend < -5 or rps_trend < -10) and action_delta < 0:
#             trend_bonus += 2.0
#         # Penalize upscaling during negative trend, and vice versa
#         if (cpu_trend < -5 or rps_trend < -10) and action_delta > 0:
#             trend_bonus -= 2.0
#         if (cpu_trend > 5 or rps_trend > 10) and action_delta < 0:
#             trend_bonus -= 2.0
#         # Penalize unnecessary scaling when trends are flat
#         if abs(action_delta) > 0 and abs(cpu_trend) < 3 and abs(rps_trend) < 5:
#             trend_bonus -= 1.0

#         # Combine into total reward
#         reward = - (latency_penalty + error_penalty + cpu_penalty +
#                     waste_penalty + scaling_penalty) + trend_bonus

#         # Bonus for meeting SLO with efficient resource use
#         if (workload['p95_latency_ms'] < self.target_latency_ms and
#             workload['error_rate_pct'] < 1.0 and
#             30 <= cpu_util <= self.max_cpu_util):
#             reward += 5.0

#         return float(reward)

    
#     def set_state(self, state_dict: Dict[str, Any]):
#         """External method to update environment state from real system"""
#         self.current_state = state_dict
        
#     def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
#         """Execute one step in the environment"""
#         if self.current_state is None:
#             raise ValueError("State not set. Call set_state() before step()")
        
#         action_delta = self._action_to_delta(action)
        
#         # Calculate reward
#         reward = self._calculate_reward(self.current_state, action_delta)
        
#         # Update episode tracking
#         self.episode_step += 1
#         terminated = self.episode_step >= self.max_episode_steps
#         truncated = False
        
#         # Prepare observation
#         obs = self._state_dict_to_obs(self.current_state)
        
#         info = {
#             'action_delta': action_delta,
#             'current_replicas': self.current_state['infra']['pods_ready'],
#             'episode_step': self.episode_step
#         }
        
#         return obs, reward, terminated, truncated, info
    
#     def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
#         """Reset environment to initial state"""
#         super().reset(seed=seed)
#         self.episode_step = 0

#         if self.current_state is None:
#             self.current_state = {
#                 'workload': {'rps': 0, 'p95_latency_ms': 100, 'error_rate_pct': 0, 'queue_length': 0},
#                 'infra': {'pods_ready': self.min_replicas, 'hpa_desired_replicas': self.min_replicas,
#                         'cpu_utilization_pct': 50, 'mem_utilization_pct': 50, 'node_count': 1,
#                         'pod_cpu_request_cores': 0.2, 'pod_cpu_limit_cores': 0.5},
#                 'time': {'minute_of_day_sin': 0, 'minute_of_day_cos': 1, 'day_of_week': 0},
#                 'scaling': {'last_action_delta': 0, 'steps_since_action': 0},
#                 'trend': {'cpu_slope': 0.0, 'rps_slope': 0.0, 'latency_slope': 0.0}  # <--- add this
#             }

#         obs = self._state_dict_to_obs(self.current_state)
#         return obs, {}



# class PPOScalingAgent:
#     """PPO agent for Kubernetes autoscaling"""
    
#     def __init__(self, 
#                  model_path: str = "models/ppo_k8s_scaler",
#                  min_replicas: int = 1,
#                  max_replicas: int = 10):
        
#         self.model_path = model_path
#         self.min_replicas = min_replicas
#         self.max_replicas = max_replicas
        
#         # Create environment
#         self.env = KubernetesScalingEnv(
#             min_replicas=min_replicas,
#             max_replicas=max_replicas
#         )
        
#         # Wrap in DummyVecEnv for SB3 compatibility
#         self.vec_env = DummyVecEnv([lambda: self.env])
        
#         # Initialize or load PPO model
#         try:
#             self.model = PPO.load(model_path, env=self.vec_env)
#             logger.info(f"Loaded existing PPO model from {model_path}")
#         except:
#             logger.info("Creating new PPO model")
#             self.model = PPO(
#                 "MultiInputPolicy",
#                 self.vec_env,
#                 learning_rate=3e-4,
#                 n_steps=2048,
#                 batch_size=64,
#                 n_epochs=10,
#                 gamma=0.99,
#                 gae_lambda=0.95,
#                 clip_range=0.2,
#                 verbose=1,
#                 tensorboard_log="./ppo_k8s_tensorboard/"
#             )
        
#         self.training_mode = False
#         self.episode_rewards = []
#         self.episode_actions = []
        
#     def predict_action(self, state_dict: Dict[str, Any]) -> Tuple[int, int]:
#         """Predict scaling action from current state"""
#         # Update environment with real state
#         self.env.set_state(state_dict)
#         obs = self.env._state_dict_to_obs(state_dict)
        
#         # Get action from PPO
#         action, _states = self.model.predict(obs, deterministic=not self.training_mode)
#         action_delta = self.env._action_to_delta(int(action))
        
#         # Clamp to valid replica range
#         current_replicas = state_dict['infra']['pods_ready']
#         new_replicas = np.clip(
#             current_replicas + action_delta,
#             self.min_replicas,
#             self.max_replicas
#         )
        
#         actual_delta = int(new_replicas - current_replicas)
        
#         logger.info(f"PPO predicted action: {action} (delta: {action_delta}) -> "
#                    f"current: {current_replicas}, new: {new_replicas}")
        
#         return int(new_replicas), actual_delta
    
#     def train_step(self, state_dict: Dict[str, Any], action_delta: int, 
#                    reward: float, next_state_dict: Dict[str, Any], done: bool):
#         """Execute one training step"""
#         if not self.training_mode:
#             return
        
#         # Update environment and perform training
#         self.env.set_state(state_dict)
#         obs = self.env._state_dict_to_obs(state_dict)
        
#         # Convert action_delta back to discrete action
#         delta_to_action = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
#         action = delta_to_action.get(action_delta, 2)
        
#         # Step environment (this will recalculate reward internally)
#         next_obs, env_reward, terminated, truncated, info = self.env.step(action)
        
#         self.episode_rewards.append(env_reward)
#         self.episode_actions.append(action_delta)
        
#         if done or terminated:
#             logger.info(f"Episode finished. Total reward: {sum(self.episode_rewards):.2f}, "
#                        f"Avg action: {np.mean(self.episode_actions):.2f}")
#             self.episode_rewards = []
#             self.episode_actions = []
    
#     def train(self, total_timesteps: int = 10000):
#         """Train the PPO model"""
#         logger.info(f"Starting PPO training for {total_timesteps} timesteps")
#         self.training_mode = True
#         self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
#         self.save_model()
#         logger.info("Training completed and model saved")
    
#     def save_model(self):
#         """Save the PPO model"""
#         self.model.save(self.model_path)
#         logger.info(f"Model saved to {self.model_path}")
    
#     def enable_training_mode(self):
#         """Enable training mode"""
#         self.training_mode = True
#         logger.info("PPO training mode enabled")
    
#     def disable_training_mode(self):
#         """Disable training mode (use for inference only)"""
#         self.training_mode = False
#         logger.info("PPO training mode disabled")
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
from typing import Optional, Dict, Any
logger = logging.getLogger(__name__)

class KubernetesScalingEnv(gym.Env):
    """Custom Gym environment for Kubernetes pod autoscaling"""

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        min_replicas: int = 1,
        max_replicas: int = 10,
        target_latency_ms: float = 1000.0,
        max_cpu_util: float = 80.0,
        seed: int = 42
    ):
        super().__init__()
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.target_latency_ms = target_latency_ms
        self.max_cpu_util = max_cpu_util
        self.n_actions = 5
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Dict({
            'workload': spaces.Box(
                low=np.array([0, 0, 0, 0], dtype=np.float32),
                high=np.array([1000, 60000, 100, 1000], dtype=np.float32),
                dtype=np.float32
            ),
            'infra': spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                high=np.array([20, 20, 100, 100, 10, 2, 2], dtype=np.float32),
                dtype=np.float32
            ),
            'time': spaces.Box(
                low=np.array([-1, -1, 0], dtype=np.float32),
                high=np.array([1, 1, 6], dtype=np.float32),
                dtype=np.float32
            ),
            'scaling': spaces.Box(
                low=np.array([-10, 0], dtype=np.float32),
                high=np.array([10, 100], dtype=np.float32),
                dtype=np.float32
            ),
            'trend': spaces.Box(
                low=np.array([-100, -100, -100], dtype=np.float32),
                high=np.array([100, 100, 100], dtype=np.float32),
                dtype=np.float32
            )
        })
        self.rng = np.random.default_rng(seed)
        self.max_episode_steps = 100
        self.reset()

    def _state_dict_to_obs(self, state_dict):
        return {
            'workload': np.array([
                state_dict['workload']['rps'],
                state_dict['workload']['p95_latency_ms'],
                state_dict['workload']['error_rate_pct'],
                state_dict['workload']['queue_length']
            ], dtype=np.float32),
            'infra': np.array([
                state_dict['infra']['pods_ready'],
                state_dict['infra']['hpa_desired_replicas'],
                state_dict['infra']['cpu_utilization_pct'],
                state_dict['infra']['mem_utilization_pct'],
                state_dict['infra']['node_count'],
                state_dict['infra']['pod_cpu_request_cores'],
                state_dict['infra']['pod_cpu_limit_cores']
            ], dtype=np.float32),
            'time': np.array([
                state_dict['time']['minute_of_day_sin'],
                state_dict['time']['minute_of_day_cos'],
                state_dict['time']['day_of_week']
            ], dtype=np.float32),
            'scaling': np.array([
                state_dict['scaling']['last_action_delta'],
                state_dict['scaling']['steps_since_action']
            ], dtype=np.float32),
            'trend': np.array([
                state_dict['trend']['cpu_slope'],
                state_dict['trend']['rps_slope'],
                state_dict['trend']['latency_slope']
            ], dtype=np.float32)
        }
    def set_state(self, state_dict: Dict[str, Any]):
        self.current_state = state_dict
    def _action_to_delta(self, action):
        return {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}[action]
    
    def _simulate_next_state(self, state, action_delta):
        # Simple simulation for next-step dynamics
        # You should replace this with your real workload/pod simulation
        state = state.copy()
        infra = state['infra']
        workload = state['workload']
        scaling = state['scaling']
        trend = state['trend']

        # Update pods based on action
        new_pods = np.clip(infra['pods_ready'] + action_delta, self.min_replicas, self.max_replicas)
        # Simulate CPU and latency change with random noise and load effect
        prev_cpu_util = infra['cpu_utilization_pct']
        cpu_util_pct = np.clip(prev_cpu_util + action_delta * self.rng.uniform(8, 15) + self.rng.normal(), 10, 98)
        prev_rps = workload['rps']
        rps = np.clip(prev_rps + int(action_delta * 80 + self.rng.normal(0, 20)), 50, 900)
        prev_latency = workload['p95_latency_ms']
        latency = np.clip(prev_latency - action_delta * self.rng.uniform(50,300) + self.rng.normal(0, 50), 80, 5000)

        # Trend calculation
        cpu_slope = cpu_util_pct - prev_cpu_util
        rps_slope = rps - prev_rps
        latency_slope = latency - prev_latency

        # Advance time feature
        step = state['scaling']['steps_since_action'] + 1
        minute_of_day_sin = self.rng.uniform(-1, 1)
        minute_of_day_cos = self.rng.uniform(-1, 1)
        day_of_week = self.rng.integers(0, 6)

        # Update state
        state['infra']['pods_ready'] = new_pods
        state['infra']['cpu_utilization_pct'] = cpu_util_pct
        state['infra']['mem_utilization_pct'] = cpu_util_pct * self.rng.uniform(0.9, 1.1)
        state['workload']['rps'] = rps
        state['workload']['p95_latency_ms'] = latency
        state['workload']['error_rate_pct'] = np.abs(latency / self.target_latency_ms - 1.)*2 + self.rng.normal(0,1)
        state['trend']['cpu_slope'] = cpu_slope
        state['trend']['rps_slope'] = rps_slope
        state['trend']['latency_slope'] = latency_slope
        state['scaling']['last_action_delta'] = action_delta
        state['scaling']['steps_since_action'] = step
        state['time']['minute_of_day_sin'] = minute_of_day_sin
        state['time']['minute_of_day_cos'] = minute_of_day_cos
        state['time']['day_of_week'] = int(day_of_week)

        return state
    
    def _calculate_reward(self, state_dict, action_delta):
        workload = state_dict['workload']
        infra = state_dict['infra']
        trend = state_dict.get('trend', {'cpu_slope': 0, 'rps_slope': 0, 'latency_slope': 0})
        cpu_trend = trend.get('cpu_slope', 0)
        rps_trend = trend.get('rps_slope', 0)
        latency_trend = trend.get('latency_slope', 0)

        latency_penalty = max(0, (workload['p95_latency_ms'] / self.target_latency_ms) - 1.0) * 5
        error_penalty = workload['error_rate_pct'] * 2.0
        cpu_util = infra['cpu_utilization_pct']
        cpu_penalty = max(0, (cpu_util - self.max_cpu_util) / 8.0)
        waste_penalty = max(0, (30 - cpu_util) / 15.0) if cpu_util < 30 and infra['pods_ready'] > self.min_replicas else 0
        scaling_penalty = abs(action_delta) * 0.2

        trend_bonus = 0
        if (cpu_trend > 2 or rps_trend > 4) and action_delta > 0:
            trend_bonus += 1.0
        if (cpu_trend < -2 or rps_trend < -4) and action_delta < 0:
            trend_bonus += 1.0
        if (cpu_trend < -2 or rps_trend < -4) and action_delta > 0:
            trend_bonus -= 1.0
        if (cpu_trend > 2 or rps_trend > 4) and action_delta < 0:
            trend_bonus -= 1.0
        if abs(action_delta) > 0 and abs(cpu_trend) < 1 and abs(rps_trend) < 3:
            trend_bonus -= 0.4

        reward = -(latency_penalty + error_penalty + cpu_penalty +
                   waste_penalty + scaling_penalty) + trend_bonus

        if (workload['p95_latency_ms'] < self.target_latency_ms and
            workload['error_rate_pct'] < 2.0 and
            30 <= cpu_util <= self.max_cpu_util):
            reward += 2.0

        return float(reward)

    def step(self, action):
        # --- apply action and update to next state ---
        action_delta = self._action_to_delta(action)
        next_state = self._simulate_next_state(self.current_state, action_delta)
        obs = self._state_dict_to_obs(next_state)
        reward = self._calculate_reward(next_state, action_delta)
        self.episode_step += 1
        terminated = self.episode_step >= self.max_episode_steps
        truncated = False
        info = {
            "action_delta": action_delta,
            "current_replicas": next_state['infra']['pods_ready'],
            "episode_step": self.episode_step
        }
        self.current_state = next_state
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_step = 0
        self.current_state = {
            'workload': {'rps': 200, 'p95_latency_ms': 500, 'error_rate_pct': 2, 'queue_length': 12},
            'infra': {'pods_ready': self.min_replicas, 'hpa_desired_replicas': self.min_replicas,
                      'cpu_utilization_pct': 50, 'mem_utilization_pct': 55, 'node_count': 1,
                      'pod_cpu_request_cores': 0.21, 'pod_cpu_limit_cores': 0.50},
            'time': {'minute_of_day_sin': 0.0, 'minute_of_day_cos': 1.0, 'day_of_week': 0},
            'scaling': {'last_action_delta': 0, 'steps_since_action': 0},
            'trend': {'cpu_slope': 0.0, 'rps_slope': 0.0, 'latency_slope': 0.0}
        }
        return self._state_dict_to_obs(self.current_state), {}

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class PPOScalingAgent:
    """PPO agent for Kubernetes autoscaling (simulation version)"""

    def __init__(
        self,
        model_path="models/ppo_k8s_scaler",
        min_replicas=1,
        max_replicas=10
    ):
        self.model_path = model_path
        self.env = KubernetesScalingEnv(
            min_replicas=min_replicas,
            max_replicas=max_replicas
        )
        self.vec_env = DummyVecEnv([lambda: self.env])
        try:
            self.model = PPO.load(model_path, env=self.vec_env)
            logger.info(f"Loaded existing PPO model from {model_path}")
        except Exception:
            logger.info("Creating new PPO model")
            self.model = PPO(
                "MultiInputPolicy",
                self.vec_env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1,
                tensorboard_log="./ppo_k8s_tensorboard/"
            )
        self.training_mode = False
    
    def enable_training_mode(self):
        self.training_mode = True
        logger.info("PPO training mode enabled")

    def disable_training_mode(self):
        self.training_mode = False
        logger.info("PPO training mode disabled")


    # In KubernetesScalingEnv:

    def train(self, total_timesteps=10000):
        logger.info(f"Starting PPO training for {total_timesteps} timesteps")
        self.training_mode = True
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=True)
        self.model.save(self.model_path)
        logger.info("Training completed and model saved")

    def predict_action(self, state_dict):
        self.env.set_state(state_dict)
        obs = self.env._state_dict_to_obs(state_dict)
        # SB3 predict wants single-environment shape
        action, _ = self.model.predict(obs, deterministic=True)
        action_delta = self.env._action_to_delta(int(action))
        current_replicas = state_dict['infra']['pods_ready']
        new_replicas = np.clip(current_replicas + action_delta, self.env.min_replicas, self.env.max_replicas)
        return int(new_replicas), int(new_replicas - current_replicas)
