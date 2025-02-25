# FroggerML
Dois modelos de machine learning, um CNN, o outro MLP, ambos para testar o desempenho no jogo Frogger de Atari

# Nome dos integrantes:
Alexander Nunes Souza \n
Isaac Levi Lira de Oliveira
Laila Maria Alves Santos
Matheus Vinicius Ramos Guimaraes
Pericles Maikon de Jesus Costa

# Informações Úteis
Para executar são necessários !pip install tensorflow "stable-baselines3[extra]" gymnasium[atari] ale-py tensorboard torch ----------------------------
Para usar GPU é necessário  !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 (versão compatível com sua GPU) ----------------------------

Ao executar frogger_cnn.py ou frogger_mlp.py serão criadas duas pastas: models e logs, se quiser visualizar o modelo aprendendo use !python -m tensorboard.main --logdir=logs para acessar os gráficos do tensorboard, você vai querer acompanhar principalmente o ep_len_mean e o ep_rew_mean, que são a duração média de um episódio e a pontuação média por episódio.


