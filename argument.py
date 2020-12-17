import argparse
import time

def get_arg():
    parser = argparse.ArgumentParser(description='Reinforcement learning for Atari games.')
    subparsers = parser.add_subparsers(title="command", dest="command")

    # Training arguments
    # General arguments
    parser_train = subparsers.add_parser(
        "train",
        help="Train a RL model.")
    parser_train.add_argument(
        "--game",
        type=str,
        required=True,
        help="Name of game, e.g., PongNoFrameskip-v4, "
             "BreakoutNoFrameskip-v4Breakout, SpaceInvadersNoFrameskip-v4,"
             "SeaquestNoFrameskip-v4, Ant-v2, Humanoid-v2, "
             "MountainCarContinuous-v0, Pendulum-v0.")
    parser_train.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="ID of gpu. None if cpu is used.")
    parser_train.add_argument(
        "--model_type",
        type=str,
        default="double_dqn",
        help="Type of model, including double_dqn, dueling_dqn, ddpg.")
    parser_train.add_argument(
        "--num_epoch",
        type=int,
        default=10000,
        help="Number of training epochs.")
    parser_train.add_argument(
        "--num_step",
        type=int,
        default=int(1e6),
        help="Number of training steps.")
    parser_train.add_argument(
        "--size_minibatch",
        type=int,
        default=32,
        help="Minibatch size for training.")
    parser_train.add_argument(
        "--size_input_image",
        type=int,
        default=84,
        help="Input image size.")
    parser_train.add_argument(
        "--len_history_frame",
        type=int,
        default=4,
        help="Number of frames used as input.")
    parser_train.add_argument(
        "--num_action_repeat",
        type=int,
        default=1,
        help="Number of frames the same action is repeated.")
    parser_train.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor.")
    parser_train.add_argument(
        "--optim_method",
        type=str,
        default="rmsprop",
        help="Optimization method used in training, e.g., sgd, momentum, rmsprop, adam.")
    parser_train.add_argument(
        "--lr",
        type=float,
        default=0.00025,
        help="Learning rate.")
    parser_train.add_argument(
        "--lr_critic",
        type=float,
        default=0.001,
        help="Learning rate for critic network.")
    parser_train.add_argument(
        "--lr_actor",
        type=float,
        default=0.0001,
        help="Learning rate for actor network.")
    ## Arguments for RMSProp
    parser_train.add_argument(
        "--rmsprop_alpha",
        type=float,
        default=0.95,
        help="Parameter alpha for RMSProp.")
    parser_train.add_argument(
        "--rmsprop_momentum",
        type=float,
        default=0.0,
        help="Momentum parameter for RMSProp.")
    parser_train.add_argument(
        "--rmsprop_epsilon",
        type=float,
        default=0.01,
        help="Parameter epsilon for RMSProp.")
    # Arguments for Adam
    parser_train.add_argument(
        "--weight_decay_critic",
        type=float,
        default=0.01,
        help="Weight decay for critic network.")
    parser_train.add_argument(
        "--weight_decay_actor",
        type=float,
        default=0,
        help="Learning rate.")
    # Arguments for saving
    parser_train.add_argument(
        "--save_path",
        type=str,
        default=".",
        help="Path for saving results, logs, models, etc.")
    parser_train.add_argument(
        "--freq_log",
        type=int,
        default=1000,
        help="Frequency for tensorboard logging.")
    parser_train.add_argument(
        "--start_time",
        type=str,
        default=time.strftime("%Y%m%d%H%M%S", time.localtime()),
        help="Time when training process starts.")
    # Arguments for DQN
    parser_train.add_argument(
        "--size_replay_memory",
        type=int,
        default=1000000,
        help="Replay memory size.")
    parser_train.add_argument(
        "--size_replay_start",
        type=int,
        default=50000,
        help="Number of samples in replay memory when learning starts.")
    parser_train.add_argument(
        "--freq_target_Q_update",
        type=int,
        default=10000,
        help="Frequency for updating parameters of target Q network.")
    parser_train.add_argument(
        "--freq_update",
        type=int,
        default=4,
        help="Frequency for updating Q network.")
    parser_train.add_argument(
        "--epsilon_init",
        type=float,
        default=1.0,
        help="Initial epsilon for epsilon-greedy exploration.")
    parser_train.add_argument(
        "--epsilon_final",
        type=float,
        default=0.1,
        help="Final epsilon for epsilon-greedy exploration.")
    parser_train.add_argument(
        "--epsilon_final_frame",
        type=int,
        default=1000000,
        help="Number of frames for epsilon to the final value.")
    # Arguments for DDPG
    parser_train.add_argument(
        "--tau",
        type=float,
        default=0.001,
        help="Soft target update.")
    parser_train.add_argument(
        "--exploration_noise",
        type=float,
        default=0.1,
        help="Soft target update.")

    # Playing arguments
    parser_play = subparsers.add_parser(
        "play",
        help="Play game with trained RL model.")
    parser_play.add_argument(
        "--game",
        type=str,
        required=True,
        help="Name of game, e.g., PongNoFrameskip-v4, "
             "BreakoutNoFrameskip-v4Breakout, SpaceInvadersNoFrameskip-v4,"
             "SeaquestNoFrameskip-v4, Ant-v2, Humanoid-v2, "
             "MountainCarContinuous-v0, Pendulum-v0.")
    parser_play.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="ID of gpu. None if cpu is used.")
    parser_play.add_argument(
        "--model_type",
        type=str,
        default="double_dqn",
        help="Type of model, including double_dqn, dueling_dqn, ddpg.")
    parser_play.add_argument(
        "--num_epoch",
        type=int,
        default=100,
        help="Number of playing epochs.")
    parser_play.add_argument(
        "--size_input_image",
        type=int,
        default=84,
        help="Input image size.")
    parser_play.add_argument(
        "--size_replay_memory",
        type=int,
        default=4,
        help="Replay memory size.")
    parser_play.add_argument(
        "--len_history_frame",
        type=int,
        default=4,
        help="Number of frames used as input.")
    parser_play.add_argument(
        "--num_action_repeat",
        type=int,
        default=1,
        help="Number of frames the same action is repeated.")
    parser_play.add_argument(
        "--optim_method",
        type=str,
        default="rmsprop",
        help="Optimization method used in training, e.g., sgd, momentum, rmsprop, adam.")
    parser_play.add_argument(
        "--load_path",
        type=str,
        default=".",
        help="Path for loading models.")
    parser_play.add_argument(
        "--save_path",
        type=str,
        default=".",
        help="Path for saving results, logs, models, etc.")
    parser_play.add_argument(
        "--freq_log",
        type=int,
        default=1000,
        help="Frequency for tensorboard logging.")
    parser_play.add_argument(
        "--start_time",
        type=str,
        default=time.strftime("%Y%m%d", time.localtime()),
        help="Time when training process starts.")

    args = parser.parse_args()

    return args
