import gymnasium as gym
import ale_py
import torch
import os
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

#Diretório para os modelos salvos, mude o nome se não quiser sobrescrever o modelo
models_dir = "models/MLP"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

#Se puder vai usar a GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

#Wrapper de punição positiva
class PunicaoFrogger(gym.Wrapper):
    def __init__(self, env):
        super(PunicaoFrogger, self).__init__(env)

    def step(self, action):
        obs, recompensa, terminado, truncado, info = self.env.step(action)
        if terminado: 
            recompensa -= 1
        return obs, recompensa, terminado, truncado, info

#Ambiente pré-processado para o Frogger
gym.register_envs(ale_py)
# Criar o ambiente normal antes de vetorizar
env = make_atari_env('ALE/Frogger-v5', n_envs=1, seed=0)
envs = env.envs  # Pegando a lista de ambientes dentro do vetor

# Aplicar a penalização no ambiente individualmente
envs[0] = PunicaoFrogger(envs[0])

# Criar um ambiente vetorizado novamente
env = VecFrameStack(env, n_stack=4) #Adiciona noção de tempo

env.reset()

#Ajuste o buffer para sua memória
TAM_BUFFER = 125000

#Modelo MLP padrão de acordo com a MlpPolicy do Stable-Baselines3
model = DQN("MlpPolicy",
             env,
             verbose=1,
             learning_rate=1e-4,
             batch_size=64,
             tensorboard_log=logdir,
             device=device,
             learning_starts=20000,
             buffer_size=TAM_BUFFER)

#Treinamento do modelo, vai salvar uma um modelo a cada 50K timesteps
#Visualize o desempenho com o tensorboard: python -m tensorboard.main --logdir=logs
TIMESTEPS = 50000
for i in range (1,21):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="MLP")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()
