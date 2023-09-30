import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from itertools import count
from collections import deque
from environment import BodyEnvironment
from wrappers import FrameStack

LEARNING_RATE = 0.001
BATCH_SIZE = 128
REPLAY_SIZE = 128000
NUM_EPISODES = 1000
HISTORY_LEN = 20
MAX_EPISODE_LEN = 200
REPORT_INTERVAL = 50
TARGET_UPDATE_INTERVAL = 10
GAMMA = 0.9

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(HISTORY_LEN, 64)
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
    env = FrameStack(env, HISTORY_LEN)
    buffer = ReplayBuffer(REPLAY_SIZE)
    loss_deque = deque(maxlen=REPORT_INTERVAL)
    reward_deque = deque(maxlen=REPORT_INTERVAL)
    bumps_deque = deque(maxlen=REPORT_INTERVAL)

    for episode in range(1, NUM_EPISODES):
        epsilon = max(0.05, 1 - episode / 200)
        state = env.reset()
        ep_loss, ep_reward, ep_bumps = 0, 0, 0

        for step in range(MAX_EPISODE_LEN):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = model(torch.tensor(state)[None]).argmax().item()

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done or step == MAX_EPISODE_LEN - 1)
            state = next_state
            ep_reward += reward
            ep_bumps += 1 if reward == -1 else 0
            if done: break

            if len(buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = (torch.tensor(x) for x in buffer.sample(BATCH_SIZE))
                state_action_values = model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
                with torch.no_grad():
                    next_state_values = target_model(next_states).max(1)[0]
                    # double dqn, doesn't work as well :(
                    # next_state_actions = model(next_states).max(1)[1]
                    # next_state_values = target_model(next_states)[torch.arange(len(next_state_actions)), next_state_actions]
                expected_state_action_values = (~dones) * next_state_values * GAMMA + rewards
                loss = F.mse_loss(state_action_values, expected_state_action_values)
                ep_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss_deque.append(ep_loss)
        reward_deque.append(ep_reward)
        bumps_deque.append(ep_bumps)

        if episode % REPORT_INTERVAL == 0:
            print(f"Episode: {episode} | Loss: {np.mean(loss_deque):.3f} | Reward: {np.mean(reward_deque):.3f} | Bumps: {np.mean(bumps_deque):.3f}")

        if episode % TARGET_UPDATE_INTERVAL == 0:
            target_model.load_state_dict(model.state_dict())


if __name__ == "__main__":
    main()
