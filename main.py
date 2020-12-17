import torch
import game
import agent
from argument import get_arg


def main():
    # Add argument parsers
    args = get_arg()

    if args.gpu != None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    # Train the model
    if args.command == 'train':
        print("Train the {} model for {}.".format(args.model_type, args.game))
        if args.model_type in ["double_dqn", "dueling_dqn"]:
            g = game.Game(args.game)
            a = agent.Agent_DQN(args, g)
            a.memory_fill()
            a.train_game()
        elif args.model_type in ["ddpg"]:
            g = game.Game(args.game)
            a = agent.Agent_DDPG(args, g)
            a.memory_fill()
            a.train_game()

    # Test the model
    if args.command == 'play':
        print("Play game {} with trained {} model.".format(args.game, args.model_type))
        if args.model_type in ["double_dqn", "dueling_dqn"]:
            g = game.Game(args.game)
            a = agent.Agent_DQN(args, g)
            a.play_game()
        if args.model_type in ["ddpg"]:
            g = game.Game(args.game)
            a = agent.Agent_DDPG(args, g)
            a.play_game()


if __name__ == '__main__':
    main()
