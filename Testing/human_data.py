import os
import threading

import keyboard

import time

from Env.game_env import DRGBarrelEnv
from Visualizer.game_visualizer import GameVisualizer

### Imports ###

def update_visualizer(visualizer: 'GameVisualizer', env: 'DRGBarrelEnv'):
    while True:
        score = env.get_score()
        kicks = env.get_kicks()
        action = env.get_last_action()

        visualizer.update_info(score=score, kicks=kicks, action=action)

        time.sleep(1 / 60)  # Update at +-60 FPS

def record_human_play():
    main_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(main_dir, os.pardir, 'Data')
    # EXTRA SAFETY
    os.makedirs(data_dir, exist_ok=True)

    file_path = os.path.join(data_dir, 'gameplay_data.pkl')

    env = DRGBarrelEnv(record_data=True)
    visualizer = GameVisualizer()

    # Start the visualizer update loop in a separate thread
    visualizer_thread = threading.Thread(target=update_visualizer, args=(visualizer, env))
    visualizer_thread.start()

    print("Recording human play. Press 'e' to execute action. Press 'ESC' to exit.")

    while True:
        if keyboard.is_pressed('e'):
            action = 1
            observation, reward, done, info = env.step(action)
            env.last_action = action
            print(f"Action taken: {action}, Reward: {reward}, Done: {done}")
            if done:
                print("Reached 100 kicks. Ending session.")
                break
        elif keyboard.is_pressed('esc'):
            print("Exiting early.")
            break

    # Close the visualizer properly
    visualizer.close()
    visualizer_thread.join()

    env.save_gameplay_data(file_path)
    print("Gameplay data saved.")

if __name__ == "__main__":
    record_human_play()