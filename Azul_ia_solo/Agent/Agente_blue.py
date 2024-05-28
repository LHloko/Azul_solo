"""
Created on Thu May 23 16:09:11 2024

@author: lbalieiro@lince.lab
"""

# Classes externas
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# Classes proprias
import Gym_Env

class Blue_agent:
    def __init__(
            self,
            learning_rate,
            epsilon,
            discount_factor,
            action_space_size,
            state_space_size
            ):

        self.epsilon = epsilon                                          # Taxa de exploração
        self.learning_rate = learning_rate                              # Taxa de aprendizado
        self.discount_factor = discount_factor                          # Fator de desconto para futuras recompensas
        self.action_space_size = action_space_size                      # Tamanho do espaço de ação
        self.q_table = np.zeros((state_space_size, action_space_size))  # Inicializa a tabela Q com zeros

        # ???
        self.training_error = []


    #recebe a observaçao
    def get_action(self, obs):
        # Escolhe uma ação de acordo com a estratégia epsilon-greedy
        if np.random.rand() < self.epsilon:
            # Exploração: escolhe uma ação aleatória
            action = np.random.randint(0, self.action_space_size)
        else:
            # Exploração: escolhe a melhor ação atual com base na tabela Q
            action = np.argmax(self.q_table[obs])

        return action

    def update(self,
               obs,
               action,
               reward,
               next_state
               ):
        # Atualiza a tabela Q com base na regra de atualização do Q-learning
        # Calcula o valor Q máximo para o próximo estado
        max_next_q_value = np.max(self.q_table[next_state])
        
        # Calcula o alvo temporal (TD target)
        td_target = reward + self.discount_factor * max_next_q_value
        
        # Calcula o erro temporal (TD error)
        td_error = td_target - self.q_table[obs][action]
        
        # Atualiza o valor Q para a ação tomada no estado atual
        self.q_table[obs][action] += self.learning_rate * td_error
