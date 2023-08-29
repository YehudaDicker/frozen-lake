from typing import Dict, Any, Tuple, Callable
from collections import defaultdict

import gym
import numpy as np
import numpy.typing as npt
from rlhw.base import BaseAgent
import random


class RandomAgent(BaseAgent):
    def __init__(self, build_env: Callable[[bool], gym.Env], params: Dict[str, Any]):
        BaseAgent.__init__(self, build_env, params)
        self.env = build_env(False)
        self.policy = np.zeros((self.env.observation_space.n, self.env.action_space.n))
    def get_action(self) -> int:
        return self.env.action_space.sample()

    def step(self, action: int) -> Tuple[npt.NDArray, float, bool]:
        next_state, reward, done, _, _ = self.env.step(action)
        return next_state, reward, done

    def learn(self):
        """A random agent does not need to learn"""
        pass

    def run(self, max_episodes: int, max_steps: int, train: bool):
        if train:
            self.env = self.build_env(render=False)
        else:
            ####Changed render to False####
            self.env = self.build_env(render=False)

        episode_rewards = dict()

        for ne in range(max_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            for nt in range(max_steps):
                action = self.get_action()
                next_state, reward, done = self.step(action)
                total_reward += reward

                if train:
                    self.learn()

                if done:
                    break

                state = next_state

            episode_rewards[ne] = total_reward
            print(f"episode {ne}: {total_reward}")

        return episode_rewards


class QLearningAgent(BaseAgent):
    def __init__(self, build_env: Callable[[bool], gym.Env], params: Dict[str, Any]):
        BaseAgent.__init__(self, build_env, params)
        ####changed self.env = build_env(render = False)
        self.env = build_env(False)
        self.render = self.params["render"]
        self.policy = np.zeros((self.env.observation_space.n, self.env.action_space.n))

        self.lr = self.params["learning_rate"]
        self.min_lr = self.params["min_learning_rate"]
        self.lr_decay = self.params["learning_rate_decay"]
        self.gamma = self.params["discount_factor"]
        self.eps = self.params["epsilon"]
        self.eps_decay = self.params["epsilon_decay"]
        self.min_eps = self.params["min_epsilon"]
    
    def get_action(self, state: npt.NDArray, train: bool) -> int:
        """Return an action given the current state"""
        # YOUR CODE HERE
        #get random number between 0-1, if less than epsilon, get random action, otherwise, get max value of q values of state
        rand = random.uniform(0,1)
        action = self.env.action_space.sample()
        if train is True:
            if self.eps > self.min_eps: 
                if rand < self.eps:
                    action = action
                else:
                    action = np.argmax(self.policy[state])
                self.eps = self.eps*self.eps_decay  
        else:
            action = np.argmax(self.policy[state])
        return action

    def step(self, action: int) -> Tuple[npt.NDArray, float, bool]:
        next_state, reward, done, _, _ = self.env.step(action)
        return next_state, reward, done

    def learn(
        self, state: npt.NDArray, action: int, reward: float, next_state: npt.NDArray
    ) -> float:
        """Update Agent's policy using the Q-learning algorithm and return the update delta"""
        # YOUR CODE HERE
        #print("policy in learn:", self.policy[state])
        if self.lr > self.min_lr:
            next_action = np.argmax(self.policy[next_state])
            delta = reward + self.gamma*self.policy[next_state, next_action] - self.policy[state, action]
            self.policy[state, action] += self.lr*delta
            self.lr = self.lr*self.lr_decay
            return delta

    def run(self, max_episodes: int, max_steps: int, train: bool):
        """Run simulations of the environment with the agent's policy"""
        if self.render and not train:
            self.env = self.build_env(render=False)
        # YOUR CODE HERE
        else:
            ####Changed render to False####
            self.env = self.build_env(render=False)

        episode_rewards = dict()

        for ne in range(max_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            for nt in range(max_steps):
                action = self.get_action(state, train)
                next_state, reward, done = self.step(action)
                total_reward += reward

                if train:
                    self.learn(state, action, reward, next_state)

                if done:
                    break

                state = next_state

            episode_rewards[ne] = total_reward
            print(f"episode {ne}: {total_reward}")
        return episode_rewards
