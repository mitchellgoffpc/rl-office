import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from itertools import count
from collections import deque
from environment import BodyEnvironment

LEARNING_RATE = 0.001
BATCH_SIZE = 64
REPLAY_SIZE = 10000
NUM_EPISODES = 10000
HISTORY_LEN = 20
MAX_EPISODE_LEN = 200
REPORT_INTERVAL = 100
TARGET_UPDATE_INTERVAL = 10
GAMMA = 0.99

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward, dtype=np.float32), np.array(next_state), np.array(done)


def main():
    model = DQN()
    target_model = DQN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    env = BodyEnvironment()
    buffer = ReplayBuffer(REPLAY_SIZE)
    loss_deque = deque(maxlen=REPORT_INTERVAL)
    reward_deque = deque(maxlen=REPORT_INTERVAL)
    bumps_deque = deque(maxlen=REPORT_INTERVAL)

    for episode in range(1, NUM_EPISODES):
        state = env.reset()
        episode_reward = 0
        episode_bumps = 0

        for step in range(MAX_EPISODE_LEN):
            if random.random() < 0.2:
                action = env.action_space.sample()
            else:
                action = model(torch.tensor(state)).argmax().item()

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done or step == MAX_EPISODE_LEN - 1)
            episode_reward += reward
            episode_bumps += 1 if reward == -1 else 0
            state = next_state
            if done: break

        if len(buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = (torch.tensor(x) for x in buffer.sample(BATCH_SIZE))

            state_action_values = model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                next_state_values = target_model(next_states).max(1)[0]
            expected_state_action_values = (~dones) * next_state_values * GAMMA + rewards
            loss = F.mse_loss(state_action_values, expected_state_action_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_deque.append(loss.item())
        reward_deque.append(episode_reward)
        bumps_deque.append(episode_bumps)

        if episode % REPORT_INTERVAL == 0:
            print(f"Episode: {episode} | Loss: {np.mean(loss_deque):.3f} | Reward: {np.mean(reward_deque):.3f} | Bumps: {np.mean(bumps_deque):.3f}")

        if episode % TARGET_UPDATE_INTERVAL == 0:
            target_model.load_state_dict(model.state_dict())


if __name__ == "__main__":
    main()
