import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from itertools import count
from torch.distributions.categorical import Categorical

INPUT_SIZE = 4
OUTPUT_SIZE = 2

NUM_EPISODES = 2000
MEMORY_SIZE = 32000
BATCH_SIZE = 128
HIDDEN_LAYER_SIZE = 128
LEARNING_RATE = 0.0003
EPSILON = 0.1
GAMMA = 0.9
TARGET_UPDATE_INTERVAL = 10
REPORT_INTERVAL = 100
RENDER_INTERVAL = 0

class Episode:
    def __init__(self):
        self.reset()

    def reset(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(tuple(args))

    def discount_rewards(self, gamma, normalize = False):
        running_add = 0.
        total_reward = 0.
        for i, (_, _, reward, *_) in list(enumerate(self.memory))[::-1]:
            running_add = running_add * gamma + reward
            total_reward += running_add

        running_add = 0.
        avg_reward = total_reward / len(self.memory) if normalize else 0.
        for i, (state, action, reward, *rest) in list(enumerate(self.memory))[::-1]:
            running_add = running_add * gamma + reward
            discounted_reward = torch.tensor([running_add - avg_reward])
            self.memory[i] = (state, action, discounted_reward, *rest)


class ReplayMemory:
    def __init__(self, capacity = None):
        self.capacity = capacity
        self.memory = []
    
    def __len__(self):
        return len(self.memory)

    def push_episode(self, episode):
        for step in episode.memory:
            self.push_transition(step)
        episode.reset()

    def push_transition(self, transition):
        if self.capacity is None or len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[random.randint(0, len(self.memory) - 1)] = transition

    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        return map(torch.stack, zip(*sample))


class CartPoleAgent(nn.Module):
    def __init__(self):
        super(CartPoleAgent, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE // 2)
        self.output = nn.Linear(HIDDEN_LAYER_SIZE // 2, OUTPUT_SIZE)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs.float()))
        x = F.relu(self.fc2(x))
        return F.softmax(self.output(x), dim=-1)


# Training

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = CartPoleAgent()
    old_agent = CartPoleAgent()
    optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    episode = Episode()
    memory = ReplayMemory(MEMORY_SIZE)
    episode_lengths, avg_episode_length = [], 0

    for episode_counter in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        state = torch.from_numpy(state)

        # Run an episode
        for step in count():
            with torch.no_grad():
                policy = agent(state.view(1, -1))
            action = Categorical(policy).sample()
            next_state, reward, done, _, stats = env.step(action.item())
            episode.push(state, action, reward)
            state = torch.from_numpy(next_state)

            if RENDER_INTERVAL and episode_counter % RENDER_INTERVAL == 0:
                env.render()
            if done: break

        # When the episode finishes, reset the environment and update the agent
        episode_lengths.append(step + 1.)
        episode.discount_rewards(GAMMA)
        memory.push_episode(episode)

        if len(memory) > BATCH_SIZE * 4:
            if episode_counter % TARGET_UPDATE_INTERVAL == 0:
                old_agent.load_state_dict(agent.state_dict())

            inputs, actions, rewards = memory.sample(BATCH_SIZE)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            
            outputs = agent(inputs)
            with torch.no_grad():
              old_outputs = old_agent(inputs)
            responsible_outputs = torch.gather(outputs, 1, actions)
            old_responsible_outputs = torch.gather(old_outputs, 1, actions).detach()

            ratio = responsible_outputs / (old_responsible_outputs + 1e-5)
            clamped_ratio = torch.clamp(ratio, 1. - EPSILON, 1. + EPSILON)
            loss = -torch.min(ratio * rewards, clamped_ratio * rewards).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Report stats every so often
        if episode_counter % REPORT_INTERVAL == 0:
            avg_episode_length = torch.tensor(episode_lengths).mean()
            print("Episode {:<4} | Avg episode length: {:.2f}".format(episode_counter, avg_episode_length))
            episode_lengths = []
