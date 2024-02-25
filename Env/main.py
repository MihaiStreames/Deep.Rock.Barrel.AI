import gym
from gym import spaces

import pyautogui
import cv2

import numpy as np
import pygetwindow as gw
import time

from pymem import Pymem
from pymem.process import module_from_name
from Utils.game_attrs import PLAYER_PTR_WIN

### Imports ###

class GameEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(GameEnv, self).__init__()
        self.pm = Pymem("FSD-Win64-Shipping.exe")
        self.game_module = module_from_name(self.pm.process_handle, "FSD-Win64-Shipping.exe").lpBaseOfDll

        self.action_space = spaces.Discrete(2)  # 0 = do nothing, 1 = kick

        self.game_window = self.find_game_window("Deep Rock Galactic")

        if self.game_window is None:
            raise Exception("Game window not found. Is the game running?")
        else:
            print("Game window found:", self.game_window.title)
        self.width, self.height = self.game_window.width, self.game_window.height

        # Dynamically set observation space based on the game window size
        CHANNELS = 3  # Assuming RGB screenshots
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, CHANNELS), dtype=np.uint8)

    def get_pointer_address(self, base, offsets):
        addr = self.pm.read_longlong(base)

        for i in offsets:
            if i is not offsets[-1]:
                addr = self.pm.read_longlong(addr + i)
            else:
                return addr + offsets[-1]

    def find_game_window(self, title_substring):
        windows = gw.getWindowsWithTitle(title_substring)
        for window in windows:
            if title_substring in window.title:
                return window
        return None

    def capture_screen(self):
        # Make sure the game window is focused and in the foreground
        self.game_window.activate()
        # Use the game window's position and size to capture just its content
        x, y = self.game_window.topleft

        screenshot = pyautogui.screenshot(region=(x, y, self.width, self.height))
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)  # Convert to BGR format for consistency

        return screenshot

    def initialize_game_state(self):
        self.score = 0
        self.previous_score = 0
        self.combo_multiplier = 1

    def execute_action(self, action):
        if action == 1:  # Kick action
            pyautogui.press('e')  # Simulate the E key press to kick

    def detect_score_change(self):
        address = self.get_pointer_address(self.game_module + PLAYER_PTR_WIN['score']['base'], PLAYER_PTR_WIN['score']['offsets'])
        self.score = self.pm.read_longlong(address)

        if self.score is None:
            print("Score not detected")
        else:
            print("Score detected:", int(self.score))

    def update_reward_and_state(self):
        self.detect_score_change()

        if self.score > self.previous_score:
            # Score increased, successful kick
            reward = 10 * self.combo_multiplier
            self.combo_multiplier *= 2  # Increase combo multiplier
        else:
            # Score did not increase, missed kick
            reward = -5  # Penalty for missing
            self.combo_multiplier = 1  # Reset combo multiplier
        self.previous_score = self.score

        return reward

    def step(self, action):
        self.execute_action(action)
        time.sleep(0.5)  # Wait for action to take effect and score to update
        reward = self.update_reward_and_state()

        observation = self.capture_screen()
        done = False  # Define your termination condition
        info = {'score': self.score, 'combo': self.combo_multiplier}

        return observation, reward, done, info

    def reset(self):
        self.initialize_game_state()
        observation = self.capture_screen()

        return observation

if __name__ == "__main__":
    # Testing if score detection works
    env = GameEnv()
    while True:
        env.detect_score_change()
        time.sleep(1)