#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:54:31 2024

@author: lbalieiro@lince.lab
"""

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Classes proprias
import Gym_Env

ambiente = "GymEnv/Azul-v1"
vec_env = make_vec_env(ambiente, n_envs=4)

# Meu ambiente
# Numero de passos
passos = 100000000

# Parallel environments


model = PPO("MlpPolicy", vec_env, verbose=0)
model.learn(total_timesteps=passos,progress_bar=True)

model.save("ppo_AZUL_chidori_v6")


'''

model = PPO.load("ppo_AZUL_chidori_v6")

obs = vec_env.reset()
for i in range(100000):
    #print('Step => ', i)
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)

    if rewards > 1:
        print(rewards)
'''

print("RENHA")