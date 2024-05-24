"""
Created on Mon May  13 21:57:49 2024

@author: lbalieiro@lince.lab
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Minhas Classes
from Game_classes_v02 import Factories
from Game_classes_v02 import States
from Game_classes_v02 import Player

class AzulEnv(gym.Env):
    metadate = {"render_modes":['ansi']}

    def __init__(self, render_mode = None):
        super(AzulEnv, self).__init__()

        # Inicializar o ambiente
        self.fab = Factories.Fabrica()  # Instanciar a fábrica
        self.players = [Player.Jogador("LH")]  # Instanciar os jogadores =+= , Jogador.Jogador("VK")
        self.dados = [self.fab, self.fab.pocket, self.players]
        self.estado = States.Estados(self.dados)

        # Definir os espaços de ação e observação
        self.action_space = spaces.MultiDiscrete([6,5,6])

        # Define o espaço de observaçao como
        # -2 (local vazio), -1 (sem ceramica), 0:4 (ceramicas postas)
        self.observation_space = spaces.Box(low=-2, high=4, shape=(1, 100), dtype=int)

        # Inicializar o estado do ambiente
        self.reset()

    '''
    Entrada: agent
    Saida: A observaçao do ambiente (fabricas-chao_de_fabrica-tabuleiros) como 
    uma matriz.
    '''
    def observe(self):
        # Obter o estado atual do jogo
        state = self.estado.get_states()

        factories       = state['fac']
        factory_floor   = state['fac-flr']
        ply_bord_01     = state['ply_01']
        #ply_bord_02     = state['ply_02']

        # Processar o estado para criar as observações
        table = np.concatenate((factories, factory_floor), axis=1)

        observations = np.concatenate((table,ply_bord_01), axis = 0) #,ply_bord_02

        #observations = np.ravel(observations)

        return observations


    '''
    Entrada: action como uma tupla com 3 valores 
    Saida: List com 'observation, reward, terminated, truncated, info'
    Implementar lógica para executar uma ação no ambiente
    '''
    def step(self, action):
        terminated = False
        truncated = False
        reward = 0
        jogada = list(action)

        # Executar a ação no ambiente
        valid_move = self.players[0].playar(self.fab ,jogada)

        # Excede
        if not valid_move:
            truncated = True
            reward -= 10 # Movimento invalido

        # Verificar se o turno acabou
        if self.estado.fim_de_turno():
            reward = self.players[0].pontuar() # Obter a recompensa do fim do turno
            # Verificar se o jogo acabou
            if self.estado.is_game_over():
                terminated = True
                reward = self.estado.fim_de_jogo()  # Obter a recompensa do fim do jogo
            # senao, reinicia o ambiente
            else:
                self.estado.iniciar_turno()

        # Atualizar o estado do jogo e retornar a observação, recompensa e sinalizadores de término
        observation = self.observe()
        info = self.get_info()

        return observation, reward, terminated, truncated, info


    '''
    Entrada: Vazia
    Saida: informaçao como um dict de quem e o primeiro jogador e se eh o ultimo round
    '''
    def get_info(self):
        first_player = None
        is_last_round = False

        if self.estado.fim_de_turno:
            for player in self.players:
                if player.me_have_minus_one_gd:
                    first_player = player.get_name()
                    break
    
        if self.estado.is_last_round_to_end():
            is_last_round = True

        info = {
            "first_player": first_player,
            "is_last_round": is_last_round
        }
    
        return info


    '''
    Entrada: Vazia
    Saida: A observaçao do estado inicial do jogo 
    Inicia um jogo do zero, com a instanciaçao de cada classe e dos jogadores 
    '''
    def reset(self, seed=None, options=None):
        # Inicializar o ambiente
        self.fab = Factories.Fabrica()  # Instanciar a fábrica
        self.players = [Player.Jogador("LH")]  # Instanciar os jogadores =+= , Jogador.Jogador("VK")

        # Dados iniciais do ambiente
        self.dados = [self.fab, self.fab.pocket, self.players]

        # Estado inicial do jogo
        self.estado = States.Estados(self.dados)

        # Retornar a observação inicial do ambiente
        return self.observe(), self.get_info()


def main():
    # Crie uma instância do ambiente personalizado
    env = AzulEnv()
    
    # Reinicie o ambiente
    observation, info = env.reset()
    max_steps = 3

    print(observation)
    print(info)

    # Execute uma ação aleatória no ambiente
    # action = env.action_space.sample()
    # observation, reward, terminated, truncated, info = env.step(action)
    terminated = False

    while terminated != True:
        a = int(input())
        b = int(input())
        c = int(input())

        if a == 10:
            break

        action = [a,b,c]
        print(action)

        observation, reward, terminated, truncated, info = env.step(action)

        # Imprima informações sobre o passo atual
        print("Observation:\n", observation)
        print("Reward:", reward)
        print("truncated:", truncated)
        print("terminated:", terminated)
        print("Info:", info)



    '''
    for step in range(max_steps):
        # Execute uma ação aleatória no ambiente
        action = env.action_space.sample()
        print(action)
        observation, reward, terminated, truncated, info = env.step(action)

        # Imprima informações sobre o passo atual
        print("Observation:\n", observation)
        print("Reward:", reward)
        print("truncated:", truncated)
        print("terminated:", terminated)
        print("Info:", info)

        # Verifique se o episódio terminou
        if truncated or terminated:
            print("Episode finished after {} steps".format(step+1))
            env.reset()
    '''


if __name__ == "__main__":
    main()
