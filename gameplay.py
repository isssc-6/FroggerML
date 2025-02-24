from stable_baselines3 import DQN
import gymnasium as gym
import ale_py
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import time

gym.register_envs(ale_py)
env = make_atari_env('ALE/Frogger-v5', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4) #Adiciona noção de tempo

#Carregue o qualquer modelo treinado na pasta models, escolha a versão
models_dir = "models/CNN1m"
model_path = f"{models_dir}/1700000.zip"
model = DQN.load(model_path, env = env)

# Avaliaçao
episode_rewards, episode_lengths = evaluate_policy(model, env, n_eval_episodes=10, render=False, return_episode_rewards=True)
media = np.mean(episode_rewards)
desvio_padrao = np.std(episode_rewards)

print(f"Mean reward: {media}")
print(f"Standard deviation of reward: {desvio_padrao}")
print(f"Episode rewards: {episode_rewards}")
print(f"Episode lengths: {episode_lengths}")

#Render para acompanhar o modelo jogando(so encerra no terminal)
obs = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")
    time.sleep(0.05)
    
