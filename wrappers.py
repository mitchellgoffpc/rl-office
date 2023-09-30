import numpy as np
import gymnasium as gym
from collections import deque

class FrameStack(gym.ObservationWrapper):
    def __init__(self, env, history_len):
        self.env = env
        self.history_len = history_len
        self.history = deque(maxlen=history_len)

    def reset(self):
        state = self.env.reset()
        for _ in range(self.history_len):
            self.history.append(state)
        return np.concatenate(self.history)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.history.append(next_state)
        return np.concatenate(self.history), reward, done, info
