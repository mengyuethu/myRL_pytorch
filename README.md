# myRL_pytorch
Pytorch implement for typical deep reinforcement learning 
algorithms, including DQN, dueling DQN, DDPG, etc. 
More other algorithms are to be added.

## Requirements

* Python3
* PyTorch
* Gym

In order to install requirements, follow:

```bash
pip install -r requirements.txt
```

## Training
Train the reinforcement learning models with 
Atari, MuJuCo, or other environments in OpenAI Gym.

### DQN
```bash
python main.py train --game PongNoFrameskip-v4 --gpu 0 --model_type double_dqn --optim_method rmsprop
```

### DDPG
```bash
python main.py train --game Humanoid-v2 --gpu 0 --model_type ddpg --optim_method adam --len_history_frame 1
```

## Testing
Play games with pretrained models.

### DQN
```bash
python main.py play --game PongNoFrameskip-v4 --gpu 0 --model_type double_dqn --optim_method rmsprop --start_time 20200101
```

### DDPG
```bash
python main.py train --game Humanoid-v2 --gpu 0 --model_type ddpg --optim_method adam --len_history_frame 1 --start_time 20201231083010
```

Note: parameter --start_time can be specified according to the filename of your pretrained model.