import os
import time

from stable_baselines3 import PPO

from Env.game_env import DRGBarrelEnv

### Imports ###

model_name = f"drgbarrels_v1_{int(time.time())}"
models_dir = f"Models/{model_name}"
log_dir = f"Logs/{model_name}"

# Create directories if they don't exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

env = DRGBarrelEnv(record_data=False)

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10000
iteration = 0

# Training loop
while True:
    print(f"Training iteration {iteration}")
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/model_{iteration}")
    iteration += 1