import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from itertools import count
from collections import deque
from environment import BodyEnvironment

BATCH_SIZE = 32
ST_BUFFER_SIZE = 100
LT_BUFFER_SIZE = 16000
HIDDEN_SIZE = 128
CLIP_EPSILON = 0.2
GAMMA = 0.9
NUM_ACTIONS = 4
TRAIN_INTERVAL = 10

class ReplayBuffer:
  def __init__(self, max_size=1000000):
    self.buffer = deque(maxlen=max_size)

  def __len__(self):
    return len(self.buffer)

  def push(self, state, action, reward):
    self.buffer.append((state, action, reward))

  def sample(self, batch_size):
    states, actions, rewards = zip(*random.sample(self.buffer, batch_size))
    return np.stack(states), np.array(actions), np.array(rewards)

class BodyModel(nn.Module):
  def __init__(self, hidden_size):
    super().__init__()
    # resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    # self.backbone = nn.Sequential(*list(resnet.children())[:-1])
    self.fc1 = nn.Linear(1, hidden_size)
    self.fc2 = nn.Linear(hidden_size, NUM_ACTIONS)
  
  def forward(self, x):
    # x = self.backbone(x).flatten(start_dim=1)
    x = self.fc1(x / 100).relu()
    x = self.fc2(x)
    return x


def train():
  device = torch.device('cuda')
  env = BodyEnvironment()
  replay_buffer = ReplayBuffer(LT_BUFFER_SIZE)
  short_term_buffer = deque()
  state = env.reset()
  # state = np.transpose(state, (2, 0, 1)).astype(np.float32) / 255.0

  model = BodyModel(HIDDEN_SIZE).to(device)
  target_model = BodyModel(HIDDEN_SIZE).to(device).eval()
  target_model.load_state_dict(model.state_dict())
  optimizer = torch.optim.Adam(model.parameters())

  rewards = []

  for step in tqdm(count()):
    # Take a step
    model.eval()
    with torch.no_grad():
      probs = model(torch.tensor(state[None]).to(device)).cpu().softmax(dim=-1)
      action = torch.distributions.Categorical(probs).sample().item()
    next_state, reward, _, _ = env.step(action)
    short_term_buffer.append((state, action, reward))
    # state = np.transpose(next_state, (2, 0, 1)).astype(np.float32) / 255.0
    rewards.append(reward)

    # REINFORCE
    R = reward
    for i in range(len(short_term_buffer)-2, 0, -1):
      R = GAMMA * R
      s, a, r = short_term_buffer[i]
      short_term_buffer[i] = (s, a, r+R)
    
    if len(short_term_buffer) > ST_BUFFER_SIZE:
      replay_buffer.push(*short_term_buffer.popleft())
    
    # Train
    if len(replay_buffer) > BATCH_SIZE and step % TRAIN_INTERVAL == 0:
      # Optimize the model
      model.train()
      states, actions, advantages = (torch.tensor(x).to(device) for x in replay_buffer.sample(BATCH_SIZE))
      advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
      
      # Compute the log probabilities for the actions taken by the model
      log_probs = model(states).log_softmax(dim=-1).gather(dim=1, index=actions[:,None]) * advantages
      with torch.no_grad():
        old_log_probs = target_model(states).log_softmax(dim=-1).gather(dim=1, index=actions[:,None])

      # Compute the clipped surrogate loss
      ratio = (log_probs - old_log_probs).exp()
      surrogate_loss = ratio * advantages
      clipped_ratio = ratio.clip(1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)
      clipped_surrogate_loss = clipped_ratio * advantages

      # Compute the policy loss as the minimum of the surrogate loss and the clipped surrogate loss
      loss = -torch.minimum(surrogate_loss, clipped_surrogate_loss).mean()
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # Print metrics every so often
    if step % 100 == 0:
      target_model.load_state_dict(model.state_dict())
      print(f'Step: {step}, Reward: {np.mean(rewards)}')
      rewards = []


if __name__ == '__main__':
  train()
