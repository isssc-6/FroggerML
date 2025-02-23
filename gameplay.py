from stable_baselines3 import DQN
import gymnasium as gym
import ale_py
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

gym.register_envs(ale_py)
env = make_atari_env('ALE/Frogger-v5', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4) #Adiciona noção de tempo

#Carregue o qualquer modelo treinado na pasta models, escolha a versão
models_dir = "models/CNN" #Exemplo
model_path = f"{models_dir}/480000.zip" #Exemplo
model = DQN.load(model_path, env = env)

# Avaliaçao
episode_rewards = evaluate_policy(model, env, n_eval_episodes=10, render=True, return_episode_rewards=True)
media = np.mean(episode_rewards[0])
desvio_padrao = np.std(episode_rewards[0])

print(f"Média das recompensas: {media}")
print(f"Desvio Padrão da recompensa: {desvio_padrao}")
print(f"Recompensas por episódio: {episode_rewards[0]}")

#Render para acompanhar o modelo jogando(so encerra no terminal)
obs = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")
    
