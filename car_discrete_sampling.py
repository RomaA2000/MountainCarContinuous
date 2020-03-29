import random
import gym
import numpy as np
from bisect import bisect_left
import statistics
import make_plot
random.seed(0)
np.random.seed(0)


def get_distribution(first, last, step):
    ans = []
    while first <= last:
        ans.append(first)
        first += step
    return ans


class QParameters:
    def __init__(self, e=0.5, a=0.3, g=0.6):
        self.eps = e
        self.alpha = a
        self.gamma = g


class QTwoDimFunctionSampling:
    def __init__(self, first, second, acts):
        self.first_dim = first
        self.second_dim = second
        self.actions = acts
        self.q_function = np.zeros([len(self.first_dim) + 1, len(self.second_dim) + 1, self.actions])

    def get_parameters_index(self, state):
        i = bisect_left(self.first_dim, state[0], 0, len(self.first_dim))
        j = bisect_left(self.second_dim, state[1], 0, len(self.second_dim))
        return i, j

class Car:
    def __init__(self, parameters, sampling, environment):
        self.q_parameters = parameters
        self.q_sampling = sampling
        self.env = environment
        self.env.seed(0)

    def get_action(self, state):
        if self.q_parameters.eps > random.uniform(0, 1):
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_sampling.q_function[state[0], state[1]])
        return action

    def max_delta(self, next_state, state, action):
        return np.max(self.q_sampling.q_function[next_state[0], next_state[1]])\
               - self.q_sampling.q_function[state[0], state[1], action]

    def learn(self, epoch, rendering):
        scores = []
        for i in range(epoch):
            flag = False
            self.q_parameters.eps -= self.q_parameters.eps / epoch
            obs = self.env.reset()
            state = self.q_sampling.get_parameters_index(obs)
            total_score = 0
            while not flag:

                if rendering:
                    self.env.render()

                action = self.get_action(state)
                next_obs, reward, flag, information = self.env.step(action)
                new_reward = reward + 100 * (abs(next_obs[1]) - abs(obs[1]))
                next_state = self.q_sampling.get_parameters_index(next_obs)
                s_f, s_s = state
                delta = new_reward + self.q_parameters.gamma * self.max_delta(next_state, state, action)
                last_delta = (1 - self.q_parameters.alpha) * self.q_sampling.q_function[s_f, s_s, action]
                self.q_sampling.q_function[s_f, s_s, action] = last_delta + self.q_parameters.alpha * delta
                state = next_state
                total_score += reward
            scores.append(total_score)
        self.env.close()
        return scores


epoch_number = 10000
eps = 0.01
alpha = 0.4
gamma = 0.8

q_parameters = QParameters(eps, alpha, gamma)

velocity = get_distribution(-0.701, 0.701, 0.01)
position = get_distribution(-1, 0.7, 0.1)

q_two_dim_func_sampling = QTwoDimFunctionSampling(position, velocity, 3)

env = gym.make('MountainCar-v0')

car = Car(q_parameters, q_two_dim_func_sampling, env)

scores = car.learn(epoch_number, False)

make_plot.make_plot(scores, epoch_number)