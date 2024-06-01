"""
Created on Wed May 29 10:25:16 2024

@author: lbalieiro@lince.lab
"""

# Classes externas
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import tensorflow as tf
from collections import deque

class DQNAgent:
    def __init__(self, model, memory, policy, nb_actions, nb_steps_warmup=10, target_model_update=1e-2):
        self.model = model
        self.memory = memory
        self.policy = policy
        self.nb_actions = nb_actions
        self.nb_steps_warmup = nb_steps_warmup
        self.target_model_update = target_model_update

        self.step = 0
        self.training = False
        self.recent_action = None
        self.recent_observation = None

    def compile(self, optimizer, metrics=[]):
        self.model.compile(optimizer=optimizer, loss='mse', metrics=metrics)

    def forward(self, observation):
        # Select an action
        if self.training:
            if self.step < self.nb_steps_warmup:
                action = self.policy.select_action(self.memory.nb_entries)
            else:
                q_values = self.model.predict_on_batch(np.array([observation]))
                action = self.policy.select_action(q_values=q_values)
            self.recent_action = action
            self.recent_observation = observation
        else:
            q_values = self.model.predict_on_batch(np.array([observation]))
            action = self.policy.select_action(q_values=q_values)
        self.step += 1
        return action

    def backward(self, reward, terminal):
        if self.training:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal)
            if terminal:
                self.step = 0

                # Train the network
                if len(self.memory) >= self.nb_steps_warmup:
                    # Sample experience from memory
                    batch = self.memory.sample(self.model.batch_size)
                    observations = batch['observations']
                    actions = batch['actions']
                    rewards = batch['rewards']
                    next_observations = batch['next_observations']
                    terminals = batch['terminals']

                    # Predict Q values for next observations
                    next_q_values = self.model.target_model.predict_on_batch(next_observations)

                    # Compute target Q values
                    target_q_values = self.policy.compute_q_values(rewards, terminals, next_q_values)

                    # Train the model
                    metrics = self.model.train_on_batch(observations, actions, target_q_values)

                    # Update target model
                    self.model.update_target_model(self.target_model_update)

                    return metrics
        return {}

