import cv2
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
PPO = False

class ReplayMemory:
    def __init__(self, capacity = None):
        self.capacity = capacity
        self.memory = []

    def __len__(self):
        return len(self.memory)

    def push_episode(self, episode):
        for step in episode:
            self.push_transition(step)

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
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE // 2)
        self.values = nn.Linear(HIDDEN_LAYER_SIZE // 2, OUTPUT_SIZE)
        self.policy = nn.Linear(HIDDEN_LAYER_SIZE // 2, OUTPUT_SIZE)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        q = self.values(x)
        p = self.policy(x).softmax(dim=-1)
        return q, p

# class CartPoleAgent(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 32, 8, stride=4)
#         self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
#         self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
#         self.fc1 = nn.Linear(64 * 7 * 7, 512)
#         self.fc2 = nn.Linear(512, OUTPUT_SIZE)
#
#     def forward(self, x):
#         x = self.conv1(x).relu()
#         x = self.conv2(x).relu()
#         x = self.conv3(x).relu().flatten(start_dim=1)
#         x = self.fc1(x).relu()
#         x = self.fc2(x).softmax(dim=-1)
#         return x

class EnvWrapper:
  def __init__(self, env): self.env = env
  def render(self): return self.env.render()
  def reset(self): return self.prepare(*self.env.reset())
  def step(self, *args): return self.prepare(*self.env.step(*args))
  def prepare(self, img, *ret):
    return torch.tensor(img), *ret
    img = cv2.resize(img, (84, 84))
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255
    return img, *ret

screen = None
def draw(img):
    import pygame
    global screen
    if not screen:
        pygame.init()
        screen = pygame.display.set_mode((img.shape[1], img.shape[0]))
    screen.blit(pygame.surfarray.make_surface(img.transpose(1, 0, 2)), (0, 0))
    pygame.display.flip()

def discount_rewards(episode, gamma, normalize=False):
    discounted_returns = 0.
    for i, (state, action, reward, *rest) in list(enumerate(episode))[::-1]:
        discounted_returns = reward + gamma * discounted_returns
        episode[i] = (state, action, torch.tensor([discounted_returns]), *rest)
    if normalize:
        rewards = torch.stack([r for _, _, r, *_ in episode], 0)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        for i, (state, action, _, *rest) in enumerate(episode):
            episode[i] = (state, action, rewards[i].clone(), *rest)


# Training

if __name__ == '__main__':
    device = torch.device('cpu')
    agent = CartPoleAgent().to(device)
    old_agent = CartPoleAgent().to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = EnvWrapper(env)
    memory = ReplayMemory(MEMORY_SIZE)
    episode_lengths, avg_episode_length, eps = [], 0, 0

    for episode_counter in range(1, NUM_EPISODES + 1):
        episode = []
        state, _ = env.reset()

        # Run an episode
        for step in count():
            with torch.no_grad():
                values, policy = agent(state[None].to(device))

            if PPO:
                action = Categorical(policy.cpu()).sample()
            else:
                eps = max(0.01, 1 - episode_counter / NUM_EPISODES)
                if random.random() < eps:
                      action = torch.tensor(env.env.action_space.sample())[None]
                else: action = values.argmax()[None]

            next_state, reward, done, truncated, stats = env.step(action.item())
            reward = -1 if done else 0
            episode.append((state, action, torch.tensor(reward), next_state, torch.tensor(done)))
            state = next_state

            if RENDER_INTERVAL and episode_counter % RENDER_INTERVAL == 0:
                draw(cv2.resize(env.render(), (84, 84)))
            if done or truncated: break

        # When the episode finishes, reset the environment and update the agent
        if PPO: discount_rewards(episode, GAMMA, normalize=False)
        memory.push_episode(episode)
        episode_lengths.append(step + 1.)

        if len(memory) > BATCH_SIZE * 4:
            if episode_counter % TARGET_UPDATE_INTERVAL == 0:
                old_agent.load_state_dict(agent.state_dict())

            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
            values, probs = agent(states.to(device))
            action_values, action_probs = values.cpu().gather(1, actions), probs.cpu().gather(1, actions)

            if PPO:
                with torch.no_grad():
                    _, old_probs = old_agent(states.to(device))
                    old_action_probs = old_probs.cpu().gather(1, actions)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
                ratio = action_probs / (old_action_probs + 1e-5)
                clamped_ratio = torch.clamp(ratio, 1. - EPSILON, 1. + EPSILON)
                policy_loss = -torch.min(ratio * rewards, clamped_ratio * rewards).mean()
                entropy_loss = (probs * probs.log()).sum(dim=1).mean()
                loss = policy_loss # + 1e-4 * entropy_loss
            else:
                with torch.no_grad():
                    next_values, _ = agent(next_states.to(device))
                best_future_values = next_values.max(dim=-1).values * ~dones
                target_values = (rewards + GAMMA * best_future_values)[:,None]
                loss = F.smooth_l1_loss(action_values, target_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Report stats every so often
        if episode_counter % REPORT_INTERVAL == 0:
            avg_episode_length = torch.tensor(episode_lengths).mean()
            print(f"Episode {episode_counter:<4} | Avg episode length: {avg_episode_length:.2f} | Epsilon: {eps:.3f}")
            episode_lengths = []
