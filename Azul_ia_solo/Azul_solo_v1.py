"""
Created on Mon May 20 19:20:03 2024

@author: lbalieiro@lince.lab
"""
# Classes externas
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# Classes proprias
import Gym_Env
from Agent import Agente_blue

# __MAIN__
def main():
    # inicializa o ambiente
    env = gym.make("GymEnv/Azul-v1")
    observation, info = env.reset()
    print("Observation:\n", observation)
    print("Info:", info)

    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    # Imprima informações sobre o passo atual
    print("Observation:\n", observation)
    print("Info:", info)
    print("Reward:", reward)
    print("truncated:", truncated)
    print("terminated:", terminated)


    ##########################################################################

    # Parâmetros de treinamento
    epsilon = 0.1
    learning_rate = 0.1
    discount_factor = 0.99
    action_space_size = 180  # Obtém o tamanho do espaço de ação do ambiente
    state_space_size = 6 ** 75 # Obtém o tamanho do espaço de observaçao do ambiente
    episodes = 1000  # Número de episódios de treinamento
    
    # Crie o agente
    agent = Agente_blue.Blue_agent(epsilon, learning_rate, discount_factor, action_space_size, state_space_size)
    
    # Treine o agente
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            agent.update(obs, action, reward, next_obs)

            total_reward += reward

            done = truncated or terminated

            obs = next_obs
        
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")
    
    print("Training finished.")




if __name__ == "__main__":
    main()
