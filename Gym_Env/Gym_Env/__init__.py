#
from gymnasium.envs.registration import register

register(
     id="GymEnv/Azul-v1",
     entry_point="Gym_Env.Azul_env:AzulEnv",
     max_episode_steps=777,
)