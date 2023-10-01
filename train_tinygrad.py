import random
import numpy as np
from itertools import count
from collections import deque
from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.nn import Linear
from tinygrad.nn.optim import Adam, get_parameters
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

class ActorCritic:
    def __init__(self):
        self.fc1 = Linear(HISTORY_LEN, 64)
        self.fc2 = Linear(64, 64)
        self.actor = Linear(64, 4)
        self.critic = Linear(64, 1)

    def __call__(self, x):
        x = x / 100
        x = self.fc1(x).tanh()
        x = self.fc2(x).tanh()
        return self.actor(x).softmax(-1), self.critic(x)


def main():
    model = ActorCritic()
    optimizer = Adam(get_parameters(model), lr=LEARNING_RATE)
    env = BodyEnvironment()
    env = FrameStack(env, HISTORY_LEN)
    buffer = ReplayBuffer(REPLAY_SIZE)

    loss_deque = deque(maxlen=REPORT_INTERVAL)
    reward_deque = deque(maxlen=REPORT_INTERVAL)
    bumps_deque = deque(maxlen=REPORT_INTERVAL)
    total_bumps = 0

    @TinyJit
    def jitmodel(x):
        probs, values = model(x)
        return probs.realize(), values.realize()

    @TinyJit
    def compute_loss(states, action_mask, rewards, next_states, dones):
        probs, state_values = model(states)
        _, next_state_values = model(next_states)
        target_values = (rewards + (1 - dones) * next_state_values[:,0].detach() * GAMMA)[:,None]
        critic_loss = (state_values - target_values).pow(2).mean()

        action_probs = (probs * action_mask).sum(-1)[:,None]
        action_probs = action_probs.clip(0.01, 1)  # probs nan out without this :(
        advantages = (target_values - state_values).detach()
        actor_loss = -(action_probs.log() * advantages).mean()

        entropy = -(probs * (probs + 1e-5).log()).sum(-1)
        entropy_loss = -ENTROPY_BETA * entropy.mean()

        loss = actor_loss + critic_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.realize()


    for episode in range(1, NUM_EPISODES):
        state = env.reset()
        ep_loss, ep_reward, ep_bumps = 0, 0, 0

        for step in range(MAX_EPISODE_LEN):
            probs, _ = jitmodel(Tensor(state[None]))
            action = np.random.choice(4, p=probs.numpy()[0])

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            ep_bumps += 1 if reward == -1 else 0
            if done: break

            if len(buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
                states, rewards, next_states, dones = (Tensor(x) for x in (states, rewards, next_states, dones))

                action_mask = np.zeros((BATCH_SIZE, 4), dtype=np.float32)
                action_mask[range(BATCH_SIZE), actions] = 1
                action_mask = Tensor(action_mask)

                loss = compute_loss(states, action_mask, rewards, next_states, dones)
                ep_loss += loss.numpy()

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
