from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from Env.main import GameEnv

# Process the footage and create a dataset for pre-training (not shown here)

env = make_vec_env(lambda: GameEnv(), n_envs=1)
model = PPO("CnnPolicy", env, verbose=1)

# Pre-train the model using your dataset (not shown here)

# Train the model
model.learn(total_timesteps=100000)
model.save("ppo_deep_rock_galactic")

# To test the model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)