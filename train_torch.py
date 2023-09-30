import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from itertools import count
from collections import deque
from environment import BodyEnvironment
from wrappers import FrameStack

LEARNING_RATE = 0.0003
BATCH_SIZE = 128
REPLAY_SIZE = 128000
NUM_EPISODES = 1000
HISTORY_LEN = 20
MAX_EPISODE_LEN = 200
REPORT_INTERVAL = 50
TARGET_UPDATE_INTERVAL = 10
GAMMA = 0.9
ENTROPY_BETA = 1e-4

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(HISTORY_LEN, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, 4)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x / 100))
        x = F.relu(self.fc2(x))
        return F.softmax(self.actor(x), dim=-1), self.critic(x)

class ReplayBuffer:
    def __init__(self, capacity):
        # self.capacity = capacity
        # self.buffer = []
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, *args):
        self.buffer.append(args)
        # if len(self.buffer) < self.capacity:
        #     self.buffer.append(args)
        # else:
        #     self.buffer[max(np.random.randint(len(self.buffer)), np.random.randint(len(self.buffer)))] = args

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward, dtype=np.float32), np.array(next_state), np.array(done)


def main():
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    env = BodyEnvironment()
    env = FrameStack(env, HISTORY_LEN)
    buffer = ReplayBuffer(REPLAY_SIZE)

    loss_deque = deque(maxlen=REPORT_INTERVAL)
    reward_deque = deque(maxlen=REPORT_INTERVAL)
    bumps_deque = deque(maxlen=REPORT_INTERVAL)
    total_bumps = 0

    for episode in range(1, NUM_EPISODES):
        state = env.reset()
        ep_loss, ep_reward, ep_bumps = 0, 0, 0

        for step in range(MAX_EPISODE_LEN):
            with torch.no_grad():
                probs, _ = model(torch.tensor(state)[None])
            action = np.random.choice(4, p=probs.numpy()[0])

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            ep_bumps += 1 if reward == -1 else 0
            if done: break

            if len(buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = (torch.tensor(x) for x in buffer.sample(BATCH_SIZE))
                probs, state_values = model(states)
                _, next_state_values = model(next_states)
                target_values = (rewards + (~dones) * next_state_values.squeeze(-1) * GAMMA)[:,None]
                critic_loss = F.mse_loss(state_values, target_values)

                action_probs = probs[range(BATCH_SIZE), actions][:,None]
                action_probs = torch.clip(action_probs, 0.01, 1)  # probs nan out without this :(
                advantages = (target_values - state_values).detach()
                actor_loss = -torch.mean(torch.log(action_probs) * advantages)

                entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=-1)
                entropy_loss = -ENTROPY_BETA * entropy.mean()

                loss = actor_loss + critic_loss + entropy_loss
                ep_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss_deque.append(ep_loss)
        reward_deque.append(ep_reward)
        bumps_deque.append(ep_bumps)
        total_bumps += ep_bumps

        if episode % REPORT_INTERVAL == 0:
            print(f"Episode: {episode} | "
                  f"Loss: {np.mean(loss_deque):.3f} | "
                  f"Avg Reward: {np.mean(reward_deque):.3f} | "
                  f"Avg Bumps: {np.mean(bumps_deque):.3f} | "
                  f"Total Bumps: {total_bumps}")


if __name__ == "__main__":
    main()
