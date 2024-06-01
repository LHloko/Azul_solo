"""
Created on Mon May 20 19:20:03 2024

@author: lbalieiro@lince.lab

# Classes externas
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# Classes projeto DQN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from tensorflow.keras.optimizers.legacy import Adam

# Classes proprias
import Gym_Env
from Agent import Agente_blue

def teste(self):
    # inicializa o ambiente
    env = gym.make("GymEnv/Azul-v1")
    
    observation, info = env.reset()

    print("Observation:\n", observation)
    print("Info:", info)

    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    # Imprima informações sobre o passo atual
    print("Observation:\n", observation)
    print("Info:", info)
    print("Reward:", reward)
    print("truncated:", truncated)
    print("terminated:", terminated)


    ##########################################################################
    '''
    # Parâmetros de treinamento
    epsilon = 0.1
    learning_rate = 0.1
    discount_factor = 0.99

    action_space_size = 180  # Obtém o tamanho do espaço de ação do ambiente

    state_space_size = 75 # Obtém o tamanho do espaço de observaçao do ambiente

    episodes = 1000  # Número de episódios de treinamento

    # Crie o agente
    agent = Agente_blue.Blue_agent(epsilon, learning_rate, discount_factor, action_space_size, state_space_size)

    agent.obs_slice_fabs(observation)



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
    '''

'''
def build_agent(model, actions):

    policy = BoltzmannQPolicy()

    memory = SequentialMemory(limit=50000, window_length=1)

    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

def build_model(states, actions):

    print('states', states)
    print('actions', actions)

    model = Sequential()


    model.add(Dense(24, activation='relu', input_shape=states))

    model.add(Dense(24, activation='relu'))

    model.add(Flatten(input_shape=states))     # Adiciona uma camada de achatamento para transformar a entrada em uma dimensão

    model.add(Dense(np.prod(actions), activation='linear', ))

    return model

# __MAIN__
def main():
    # inicializa o ambiente
    env = gym.make("GymEnv/Azul-v1")

    # ???

    states  = env.observation_space.shape
    actions = env.action_space.nvec.shape[-1]

    print(states)
    print('act', actions)

    model = build_model(states, actions)

    model.summary()

    dqn = build_agent(model, 180)

    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

    dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

    return


def build_model(states, actions):
     model = Sequential()

     model.add(Dense(24, activation='relu', input_shape=states))

     #model.add(Dense(24, activation='relu', ))

     model.add(Dense(actions, activation='linear'))

     return model

def build_agent(model, actions):
    print(actions)

    policy = BoltzmannQPolicy()

    memory = SequentialMemory(limit=50000, window_length=1)

    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  nb_actions=180, nb_steps_warmup=10, target_model_update=1e-2)

    return dqn

def main():
    # Construo um ambiente
    env = gym.make("GymEnv/Azul-v1")

    obs, info = env.reset()

    # Pego o formato da observaçao
    states  = env.observation_space.shape

    # Pego a quantidade de açoes
    actions = env.action_space.nvec     # [6,5,6]
    actions = np.prod(actions)          # 180

    # Construo o modelo
    model = build_model(states, actions)

    # Imprimo o sumario
    model.summary()

    # Constroi o agente
    dqn = build_agent(model, actions)

    # Compila o agente com a metrica de peso
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

    # Roda o treinamento
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
    '''

def main():


if __name__ == "__main__":
    main()
"""

"""
Created on Mon May 20 19:20:03 2024

@author: lbalieiro@lince.lab
"""

# Classes externas
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# Classes projeto RL
#from ray.rllib.agents.ppo import PPOTrainer
import ray

# Classes proprias
import Gym_Env

def main():
    # inicializa o ambiente
    # env = gym.make("GymEnv/Azul-v1")
    
    ''' 
    observation, info = env.reset()

    print("Observation:\n", observation)
    print("Info:", info)

    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    # Imprima informações sobre o passo atual
    print("Observation:\n", observation)
    print("Info:", info)
    print("Reward:", reward)
    print("truncated:", truncated)
    print("terminated:", terminated)
    '''

    #
    # Initialize Ray
    ray.init()
    
    # Create the environment
    env = gym.make("CartPole-v1")
    
    # Create a PPO trainer
    trainer = PPOTrainer(
        env=env,
        config={
            "num_workers": 1,
            "env_config": {},
            "lambda": 0.95,
            "num_gpus": 0,
            "model": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu"
            },
            "train_batch_size": 2048,
            "lr": 0.0003,
            "gamma": 0.99,
            "num_policy_timesteps": 2000000,
            "num_env_steps_sampled_per_iteration": 2000,
            "compression_mode": "none"
        }
    )
    
    # Train the agent
    result = trainer.train()
    
    # Get the number of steps taken during training
    num_steps = result["episode_len_mean"] * result["timesteps_total"]
    
    # Evaluate the trained agent
    episode_rewards = []
    for _ in range(10):
        episode_reward = 0
        done = False
        state = env.reset()
        while not done:
            action = trainer.compute_action(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)
    
    # Print the results
    print(f"Training took {num_steps} steps.")
    print(f"Mean episode reward: {sum(episode_rewards) / len(episode_rewards)}")
    
    # Clean up
    env.close()
    ray.shutdown()



if __name__ == "__main__":
    main()