#
from gymnasium.envs.registration import register

register(
     id="Azul-v0",
     entry_point="Gym_env.Azul_env:Azul_v0",
     max_episode_steps=777,
)