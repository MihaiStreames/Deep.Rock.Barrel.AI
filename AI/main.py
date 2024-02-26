from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from Env.game_env import GameEnv

### Imports ###

env = make_vec_env(lambda: GameEnv(), n_envs=1)
model = PPO("CnnPolicy", env, verbose=1)

model.learn(total_timesteps=10000)
model.save("ppo_drgbarrels_v1")

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    # env.render() - not yet, will use the GameVisualizer class

env.close()