import gym
from gym import spaces

import pyautogui
import cv2

import numpy as np
import pygetwindow as gw
import time

from pymem import Pymem
from pymem.process import module_from_name

from threading import Thread

from Utils.game_attrs import PTR_DICT

### Imports ###

class GameEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(GameEnv, self).__init__()

        # Memory stuff
        self.pm = Pymem("FSD-Win64-Shipping.exe")
        self.game_module = module_from_name(self.pm.process_handle, "FSD-Win64-Shipping.exe").lpBaseOfDll

        self.action_space = spaces.Discrete(2)  # 0 = do nothing, 1 = kick

        self.game_window = self.find_game_window("Deep Rock Galactic")

        if self.game_window is None:
            raise Exception("Game window not found. Is the game running?")
        else:
            print("Game window found:", self.game_window.title)
        self.width, self.height = self.game_window.width, self.game_window.height

        CHANNELS = 3
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, CHANNELS), dtype=np.uint8)
        self.initialize_game_state()

    def initialize_game_state(self):
        self.score = 0
        self.previous_score = 0
        self.kicks = 0

        self.done = False

        self.deduct_thread = Thread(target=self.deduct_score)
        self.deduct_thread.start()

    def get_pointer_address(self, base, offsets):
        addr = self.pm.read_longlong(base)

        for i in offsets:
            if i is not offsets[-1]:
                addr = self.pm.read_longlong(addr + i)
            else:
                return addr + offsets[-1]

    def memory_tracking(self):
        # Track score
        score_address = self.get_pointer_address(self.game_module + PTR_DICT['score']['base'], PTR_DICT['score']['offsets'])
        self.score = self.pm.read_longlong(score_address)

        # Track kicks
        kicks_address = self.get_pointer_address(self.game_module + PTR_DICT['kicks']['base'], PTR_DICT['kicks']['offsets'])
        memory_kicks = self.pm.read_longlong(kicks_address)

        if self.score is None or memory_kicks is None:
            # TODO: Fallback addresses (PTR_DICT['title']['fallback'] -> list of bases)
            print("Fallback: Score or kicks not detected using main pointer.")

        if memory_kicks is not None:
            if memory_kicks != self.kicks:
                print(f"Discrepancy detected: internal kicks ({self.kicks}) vs memory kicks ({memory_kicks}), setting kicks to {memory_kicks}")
                self.kicks = memory_kicks

        # Update self.score and self.kicks if no discrepancy or after adjustment
        if self.score is not None:
            print("Score detected:", int(self.score))
        if self.kicks is not None:
            print("Kicks detected:", int(self.kicks), "mem:", int(memory_kicks))

    def find_game_window(self, title):
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

    def deduct_score(self):
        while not self.done:
            time.sleep(1)
            self.score -= 1
            if self.score < 0:
                self.score = 0

    def execute_action(self, action):
        if action == 1:
            pyautogui.press('e')
            self.kicks += 1

    def update_reward_and_state(self):
        self.memory_tracking()

        if self.score > self.previous_score:
            reward = 10 * self.combo_multiplier
            self.combo_multiplier *= 2
        else:
            reward = -5
            self.combo_multiplier = 1
        self.previous_score = self.score

        return reward

    def step(self, action):
        self.execute_action(action)
        time.sleep(0.5)

        reward = self.update_reward_and_state()
        observation = self.capture_screen()

        done = self.kicks >= 100  # Terminate when kicks reach 100
        info = {'score': self.score, 'kicks': self.kicks, 'combo': self.combo_multiplier}

        return observation, reward, done, info

    def reset(self):
        self.initialize_game_state()
        observation = self.capture_screen()

        return observation

if __name__ == "__main__":
    # Testing if detection works
    env = GameEnv()
    while True:
        env.memory_tracking()
        time.sleep(1)