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

class Blue_Q_agent:
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
        # OK Recortar os 20 primeiros elementos
        # OK Separar em 5 tuplas de 4 elementos
        # OK Gerar uma tupla com as fabricas que nao estejam vazias
            # Sortear um elemento dali para ser a fabrica escolhida
                # Sortear uma peça ali dentro para ser a peça escolhida
                    # Criar e chamar uma funçao que retorna as linhas adjacentes
                    # que podem receber aquela ceramica

        # Escolhe uma ação de acordo com a estratégia epsilon-greedy
        if np.random.rand() < self.epsilon:
            # Exploração: escolhe uma ação aleatória
            action = np.random.randint(0, self.action_space_size)
        else:
            # Exploração: escolhe a melhor ação atual com base na tabela Q
            action = np.argmax(self.q_table[obs])

        print('action get_action = ',action)
        return action

    # ???

    '''
    Entrada: Vetor de observaçao contendo 75 elementos
    Saida: Uma list contendo quais as fabricas nao vazias 
    '''
    def obs_slice_fabs(self, obs):
        # Recorta os 20 primeiros elementos do vetor de observaçao
        fabricas = obs[:20]

        # Re-separa as fabricas
        fabs = [fabricas[i:i+4] for i in range(0, len(fabricas), 4)]

        # Verifica as fabricas nao vazias e armazena seu indice em uma list
        list_fabs = []
        for i,fa in enumerate(fabs):
            if fa[0] != -1:
                list_fabs.append(i)

        return list_fabs



    def obs_slice_floor(self, obs):


        pass

    def obs_slice_lines(self, obs):


        pass

    def slice_fab(self, fabs):


        pass














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
