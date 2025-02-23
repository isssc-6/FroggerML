# FroggerML
Dois modelos de machine learning, um CNN, o outro MLP, ambos para testar o desempenho no jogo Frogger de Atari

Ao executar frogger_cnn.py ou frogger_mlp.py serão criadas duas pastas: models e logs, se quiser visualizar o modelo aprendendo digite python -m tensorboard.main --logdir=logs para acessar os gráficos do tensorboard, você vai querer acompanhar principalmente o ep_len_mean e o ep_rew_mean, que são a duração média de um episódio e a pontuação média por episódio.

link para os gráficos de um modelo treinado
https://docs.google.com/document/d/1rhEXKXctke8BAWo1RtsgqAxXPSZoX2_YDSZxGyoVWoQ/edit?usp=sharing
