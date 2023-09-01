import random
import numpy as np
from tqdm import tqdm
from itertools import count
from collections import deque

from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.nn import Linear
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict
from efficientnet import EfficientNet
from environment import BodyEnvironment

BATCH_SIZE = 32
BUFFER_SIZE = 100
HIDDEN_SIZE = 128
CLIP_EPSILON = 0.2
GAMMA = 0.99
NUM_ACTIONS = 4

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

class BodyModel:
  def __init__(self, hidden_size):
    self.backbone = EfficientNet(0, has_fc_output=False)
    self.backbone.load_from_pretrained()
    self.fc1 = Linear(1280, hidden_size)
    self.fc2 = Linear(hidden_size, NUM_ACTIONS)
  
  def __call__(self, x:Tensor) -> Tensor:
    x = self.backbone(x)
    x = self.fc1(x).relu()
    x = self.fc2(x)
    return x


def train():
  env = BodyEnvironment()
  replay_buffer = ReplayBuffer()
  short_term_buffer = deque()
  state = env.reset()
  state = np.transpose(state, (2, 0, 1))

  _model = BodyModel(HIDDEN_SIZE)
  _target_model = BodyModel(HIDDEN_SIZE)
  load_state_dict(_target_model, get_state_dict(_model))
  optimizer = Adam(get_parameters(_model))

  @TinyJit
  def model(x):
    return _model(x).realize()
  
  # @TinyJit
  def compute_loss(states, actions, advantages):
    # Compute the log probabilities for the actions taken by the model
    log_probs = _model(states).log_softmax().gather(actions[:,None], dim=1) * advantages
    Tensor.no_grad = True
    old_log_probs = _target_model(states).log_softmax(actions).gather(actions[:,None], dim=1)
    Tensor.no_grad = False

    # Compute the clipped surrogate loss
    ratio = (log_probs - old_log_probs).exp()
    surrogate_loss = ratio * advantages
    clipped_ratio = ratio.clip(1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)
    clipped_surrogate_loss = clipped_ratio * advantages

    # Compute the policy loss as the minimum of the surrogate loss and the clipped surrogate loss
    return -surrogate_loss.minimum(clipped_surrogate_loss).mean().realize()

  # while True:
  for _ in tqdm(count()):
    # Take a step
    action = model(Tensor(state[None])).numpy().argmax()
    next_state, reward, _, _ = env.step(action)
    next_state = np.transpose(next_state, (2, 0, 1))
    short_term_buffer.append((state, action, reward))
    state = next_state

    # REINFORCE
    R = reward
    for i in range(len(short_term_buffer)-2, 0, -1):
      R = GAMMA * R
      s, a, r = short_term_buffer[i]
      short_term_buffer[i] = (s, a, r+R)
    
    if len(short_term_buffer) > BUFFER_SIZE:
      replay_buffer.push(*short_term_buffer.popleft())
    
    # Train
    if len(replay_buffer) > BATCH_SIZE:
      # Optimize the model
      states, actions, advantages = (Tensor(x) for x in replay_buffer.sample(BATCH_SIZE))
      loss = compute_loss(states, actions, advantages)
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


if __name__ == '__main__':
  train()