import random
from collections import deque
import torch
from torch import optim
from torch import distributions
from torch import nn
import gym
import numpy as np
import make_plot

random.seed(0)
np.random.seed(0)

def advantage_norm(A):
    std = 1e-4 + A.std() if len(A) > 0 else 1
    adv = (A - A.mean()) / std
    return adv

def entropyLoss(prob):
    return (prob * torch.log(prob)).sum(1).mean()


def compute_returns(next_value, rewards, gamma=0.9):
    r = next_value
    returns = deque()
    for step in reversed(range(len(rewards))):
        r = rewards[step] + gamma * r
        returns.appendleft(r)
    return list(returns)

size = 4

def make_sequence(input_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, size),
        nn.ReLU(),
        nn.Linear(size, size),
        nn.Tanh(),
        nn.Linear(size, output_size))


class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.forwarding = make_sequence(self.state_size, 1)

    def forward(self, state):
        out = self.forwarding.forward(state)
        return out


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.forwarding = make_sequence(self.state_size, action_size)

    def forward(self, state):
        out = self.forwarding.forward(state)
        out = nn.functional.softmax(out, dim=0)
        distribution = distributions.Categorical(out)
        return out, distribution


env = gym.make('MountainCar-v0').env
device = torch.device('cpu')

actor = Actor(2, 3).to(device)
critic = Critic(2).to(device)
optimizerA = optim.Adam(actor.parameters())
optimizerC = optim.Adam(critic.parameters())


n_iters = 100

scores = []
for iter in range(n_iters):
    state = env.reset()
    probs = []
    log_probs = []
    values = []
    my_rewards = []
    rewards = []
    i = 0
    while True:
        #env.render()
        state = torch.FloatTensor(state)
        prob, dist = actor(state)
        value = critic(state)
        action = dist.sample()
        next_state, reward, done, _ = env.step(action.item())
        new_reward = reward + 100 * (abs(next_state[1])-abs(state[1]))
        log_prob = dist.log_prob(action).unsqueeze(0)
        probs.append(dist.probs.unsqueeze(0))
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.tensor([reward], dtype=torch.float))
        my_rewards.append(torch.tensor([new_reward], dtype=torch.float))
        state = next_state
        if i % 10000 == 0:
            print('10000 iterations')
        if done or i > 100000:
            print(done)
            print('Iteration: {}'.format(i))
            break
        i += 1
    next_state = torch.FloatTensor(next_state)
    next_value = critic(next_state)
    returns = compute_returns(next_value, my_rewards)
    log_probs = torch.cat(log_probs)
    probs = torch.cat(probs)
    entropy = entropyLoss(probs)
    returns = torch.cat(returns).detach()
    values = torch.cat(values)
    advantage = advantage_norm(returns - values)
    actor_loss = -(log_probs * advantage.detach()).mean()
    actor_loss += entropy * 0.1
    critic_loss = advantage.pow(2).mean()
    optimizerA.zero_grad()
    optimizerC.zero_grad()
    actor_loss.backward()
    critic_loss.backward()
    optimizerA.step()
    optimizerC.step()
    scores.append(torch.tensor(rewards).sum().item())

make_plot.make_plot(scores, n_iters)