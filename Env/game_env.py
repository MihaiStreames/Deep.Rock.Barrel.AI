import gym
from gym import spaces

import pyautogui
import pickle
import cv2

import numpy as np
import pygetwindow as gw
import time

from Utils.mem_extract import MemExtract

from threading import Thread

### Imports ###

class DRGBarrelEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DRGBarrelEnv, self).__init__()

        self.action_space = spaces.Discrete(2)  # 0 = do nothing, 1 = kick

        self.game_window = self.find_game_window("Deep Rock Galactic")
        self.mem = MemExtract("FSD-Win64-Shipping.exe")

        if self.game_window is None:
            raise Exception("Game window not found. Is the game running?")
        else:
            print("Game window found:", self.game_window.title)
        self.width, self.height = self.game_window.width, self.game_window.height

        CHANNELS = 3
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, CHANNELS), dtype=np.uint8)

        self.is_first_reset = True

        self.score = 0
        self.previous_score = 0
        self.kicks = 0
        self.last_kick_time = 0
        self.combo_multiplier = 1

        self.last_action = None
        self.done = False

        self.observation_buffer = None

        self.capture_thread = Thread(target=self.continuous_capture)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        self.deduct_thread = Thread(target=self.deduct_score)
        self.deduct_thread.daemon = True
        self.deduct_thread.start()

    def get_score(self) -> int:
        return self.score

    def get_kicks(self) -> int:
        return self.kicks

    def get_last_action(self):
        return self.last_action

    def find_game_window(self, title: str):
        windows = gw.getWindowsWithTitle(title)

        for window in windows:
            if title in window.title:
                return window
        return None

    def capture_screen(self):
        self.game_window.activate()

        x, y = self.game_window.topleft

        screenshot = pyautogui.screenshot(region=(x, y, self.width, self.height))
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)  # Convert to BGR format for consistency

        return screenshot

    def continuous_capture(self):
        while not self.done:
            self.observation_buffer = self.capture_screen()
            time.sleep(1 / 60)

    def deduct_score(self):
        while not self.done:
            time.sleep(5)
            self.score -= 1
            if self.score < 0:
                self.score = 0

    def execute_action(self, action):
        if action == 1:
            pyautogui.press('e')
            self.kicks += 1
        self.last_action = action

    def update_reward_and_state(self) -> int:
        score_change = self.score - self.previous_score

        if score_change > 0:
            reward = 10 * score_change * self.combo_multiplier
            self.combo_multiplier = min(self.combo_multiplier * 2, 10)
        else:
            reward = 1
            self.combo_multiplier = 1

        self.previous_score = self.score
        return reward

    def step(self, action):
        self.execute_action(action)
        time.sleep(0.5)

        # Memory stuff :nerd:
        memory_data = self.mem.extract_memory()

        self.score = memory_data['score']

        if memory_data['kicks'] is not None and self.kicks != memory_data['kicks']:
            print(f"Discrepancy in kicks corrected: Env({self.kicks}) vs Memory({memory_data['kicks']})")
            self.kicks = memory_data['kicks']

        reward = self.update_reward_and_state()
        print(f"Reward: {reward}")

        observation = self.observation_buffer

        done = self.kicks >= 100
        info = {'score': self.score, 'kicks': self.kicks, 'combo': self.combo_multiplier}
        print(f"Info: {info}")

        return observation, reward, done, info

    def reset(self):
        self.done = True

        if self.capture_thread.is_alive():
            self.capture_thread.join()
        if self.deduct_thread.is_alive():
            self.deduct_thread.join()

        self.score = 0
        self.previous_score = 0
        self.kicks = 0
        self.combo_multiplier = 1
        self.last_action = None
        self.done = False
        self.observation_buffer = None

        if not self.is_first_reset:
            input("Press Enter in the terminal to continue with the next episode...")
        else:
            print("First reset, continuing without waiting for input...")
            self.is_first_reset = False

        self.start_threads()

        observation = self.capture_screen()
        return observation

    def start_threads(self):
        self.capture_thread = Thread(target=self.continuous_capture)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        self.deduct_thread = Thread(target=self.deduct_score)
        self.deduct_thread.daemon = True
        self.deduct_thread.start()

    def close(self):
        self.done = True
        self.capture_thread.join()
        self.deduct_thread.join()

        cv2.destroyAllWindows()