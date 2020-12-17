import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random
import matplotlib.pyplot as plt
import model
import memory
import os
import logging
import time
import csv


class Agent(object):
    def __init__(self, args, game):
        super(Agent, self).__init__()
        self.args = args
        self.game = game
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

    def train_game(self):
        pass

    def play_game(self):
        pass

    def __action_selection(self, phi):
        pass

    def __save_statistic(self, label, nums, save_path=None):
        pass


class Agent_DQN(Agent):
    def __init__(self, args, game):
        super(Agent_DQN, self).__init__(args, game)

        if self.args.model_type == 'double_dqn':
            self.net = model.Double_DQN(args.size_input_image,
                                        args.len_history_frame,
                                        game.num_action).type(self.dtype)
            self.net_target = model.Double_DQN(args.size_input_image,
                                               args.len_history_frame,
                                               game.num_action).type(self.dtype)
            self.net_target.load_state_dict(self.net.state_dict())
            self.mem = memory.MemoryReplay(args.size_replay_memory,
                                           (args.size_input_image, args.size_input_image),
                                           args.len_history_frame)
        elif self.args.model_type == 'dueling_dqn':
            self.net = model.Dueling_DQN(args.size_input_image,
                                         args.len_history_frame,
                                         game.num_action).type(self.dtype)
            self.net_target = model.Dueling_DQN(args.size_input_image,
                                                args.len_history_frame,
                                                game.num_action).type(self.dtype)
            self.net_target.load_state_dict(self.net.state_dict())
            self.mem = memory.MemoryReplay(args.size_replay_memory,
                                           (args.size_input_image, args.size_input_image),
                                           args.len_history_frame)

        self.action_space = game.action_space
        self.cnt_action_repeat = 0

        if args.command == "train":
            self.epsilon = args.epsilon_init
            self.cnt_model_update = 0
            self.cnt_iter = 0
            self.save_path_results = os.path.join(args.save_path, "results")
            self.save_path_logs = os.path.join(args.save_path, "logs")
            self.save_path_models = os.path.join(args.save_path, "models")
        elif args.command == "play":
            self.load_path_models = os.path.join(args.load_path, "models")

        self.writer = SummaryWriter()

    def memory_fill(self):
        print("==========Initialize the replay memory===========")

        obs_pre = self.game.env.reset()
        obs_pre = obs_pre[:, :, -1]

        self.mem.put_obs(obs_pre)
        for _ in range(int(self.args.size_replay_start/self.args.num_action_repeat)):
            a = random.sample(self.action_space, 1)[0]
            while self.cnt_action_repeat < self.args.num_action_repeat:
                obs, r, done, info = self.game.env.step(a)
                obs = obs[:, :, -1]
                r_normalized = np.sign(r)

                self.mem.put_effect(a, r_normalized, done)
                if done:
                    obs_pre = self.game.env.reset()
                    obs_pre = obs_pre[:, :, -1]
                else:
                    obs_pre = obs
                self.mem.put_obs(obs_pre)
                self.cnt_action_repeat += 1
            self.cnt_action_repeat = 0

    def train_game(self):
        print("==========Start training===========")
        if self.args.optim_method == "rmsprop":
            opt = optim.RMSprop(self.net.parameters(), lr=self.args.lr, alpha=self.args.rmsprop_alpha,
                                eps=self.args.rmsprop_epsilon)
        score = []
        score_mean = []
        score_mean_max = -float('inf')
        q_value = []
        cnt_time_step = 0

        # Initiate logger
        logger = logging.getLogger()
        if not os.path.exists(self.save_path_logs):
            os.makedirs(self.save_path_logs)
        file_name_logger = os.path.join(self.save_path_logs,
                                        "train_info_{}_{}_{}_{}.log".format(
                                            self.args.game,
                                            self.args.model_type,
                                            self.args.optim_method,
                                            self.args.start_time))
        if os.path.exists(file_name_logger):
            os.remove(file_name_logger)
        handler_file = logging.FileHandler(file_name_logger)
        handler_stream = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: \n %(message)s')
        handler_file.setFormatter(formatter)
        handler_stream.setFormatter(formatter)
        logger.addHandler(handler_file)
        logger.addHandler(handler_stream)
        logger.setLevel(logging.INFO)

        for epoch in range(self.args.num_epoch):
            obs = self.game.env.reset()
            obs = obs[:, :, -1]

            self.mem.put_obs(obs)
            done = False
            score_tmp = 0
            q_tmp = 0
            a = 0
            while not done:
                opt.zero_grad()
                cnt_time_step += 1

                # Select a new action every self.num_action_repeat frames
                if self.cnt_action_repeat == 0:
                    phi = self.mem.encode_frame()
                    a = self.__action_selection(phi/255.0)
                    self.cnt_model_update = (self.cnt_model_update + 1) % self.args.freq_update
                self.cnt_action_repeat = (self.cnt_action_repeat + 1) % self.args.num_action_repeat

                # Execute the action
                obs_next, r, done, info = self.game.env.step(a)
                obs_next = obs_next[:, :, -1]
                r_normalized = np.sign(r)

                self.mem.put_effect(a, r_normalized, done)
                obs = obs_next
                self.mem.put_obs(obs)
                score_tmp += float(r)

                # Update Q network every self.freq_update times of action selections
                if self.cnt_model_update == 0:
                    # Sample from replay memory
                    phi_b, a_b, r_b, phi_next_b, done_b = self.mem.sample(self.args.size_minibatch)
                    phi_batch = torch.from_numpy(phi_b).type(self.dtype) / 255.0
                    a_batch = torch.from_numpy(a_b).type(self.dlongtype)
                    r_batch = torch.from_numpy(r_b).type(self.dtype)
                    phi_next_batch = torch.from_numpy(phi_next_b).type(self.dtype) / 255.0
                    done_batch = torch.from_numpy(done_b.astype(float)).type(self.dtype)

                    # Compute loss function
                    loss = self.__get_loss(phi_batch, a_batch, r_batch, phi_next_batch, done_batch)
                    loss.backward(torch.ones_like(loss).type(self.dtype))
                    opt.step()
                    self.cnt_iter += 1

                    # Calculate Q values for the sampled states in order to illustrate
                    q_tmp += float(self.net(phi_next_batch).mean().cpu())

                    # Update the target Q network every self.freq_target_Q_update times of Q network updates
                    if self.cnt_iter == self.args.freq_target_Q_update:
                        self.net_target.load_state_dict(self.net.state_dict())
                        self.cnt_iter = 0

                if cnt_time_step % self.args.freq_log == 0:
                    self.writer.add_graph(self.net, phi_batch)
                    self.writer.add_scalar('loss', loss.mean(), cnt_time_step)

            score.append(score_tmp)
            score_mean.append(np.mean(score[-100:]))
            if score_mean[-1] > score_mean_max:
                if not os.path.exists(self.save_path_models):
                    os.makedirs((self.save_path_models))
                file_name_model = os.path.join(self.save_path_models,
                                               "{}_{}_{}_{}.pt".format(
                                                   self.args.game,
                                                   self.args.model_type,
                                                   self.args.optim_method,
                                                   self.args.start_time))
                torch.save(self.net.state_dict(), file_name_model)
            score_mean_max = max(score_mean_max, score_mean[-1])
            q_value.append(q_tmp)
            self.__save_statistic(['Training_epoch', 'Average_score', 'Average_action_value'],
                                   [epoch, score_mean, q_value], self.save_path_results)
            logging.info('Episode: {0} \n timestep {1} \n mean score {2:6f} \n best mean score {3:6f} \n epsilon: {4:6f}'
                  .format(epoch, cnt_time_step, score_mean[-1], score_mean_max, self.epsilon))

    def play_game(self):
        print("==========Start playing===========")
        file_name_model = os.path.join(self.load_path_models,
                                       "{}_{}_{}_{}.pt".format(
                                           self.args.game,
                                           self.args.model_type,
                                           self.args.optim_method,
                                           self.args.start_time))
        print(file_name_model)
        try:
            self.net.load_state_dict(torch.load(file_name_model))
        except FileNotFoundError:
            print('File does not exist!')

        score = []
        score_max = -float('inf')
        cnt_time_step = 0

        for epoch in range(self.args.num_epoch):
            obs_pre = self.game.env.reset()
            obs_pre = obs_pre[:, :, -1]
            for _ in range(self.args.len_history_frame * self.args.num_action_repeat):
                a = random.sample(self.action_space, 1)[0]
                while self.cnt_action_repeat < self.args.num_action_repeat:
                    obs, r, done, info = self.game.env.step(a)
                    obs = obs[:, :, -1]
                    r_normalized = np.sign(r)
                    self.mem.put_obs(obs_pre)
                    self.mem.put_effect(a, r_normalized, done)
                    if done:
                        obs_pre = self.game.env.reset()
                        obs_pre = obs_pre[:, :, -1]
                    else:
                        obs_pre = obs
                    self.cnt_action_repeat += 1
                self.cnt_action_repeat = 0

            done = False
            score_tmp = 0
            a = 0
            while not done:
                cnt_time_step += 1

                # Select a new action every self.num_action_repeat frames
                if self.cnt_action_repeat == 0:
                    self.mem.put_obs(obs)
                    phi = self.mem.encode_frame()
                    a = self.__action_selection(phi/255.0)
                self.cnt_action_repeat = (self.cnt_action_repeat + 1) % self.args.num_action_repeat

                # Execute the action
                self.game.env.render()
                time.sleep(0.01)
                obs_next, r, done, info = self.game.env.step(a)
                obs_next = obs_next[:, :, -1]
                r_normalized = np.sign(r)

                # self.mem.put_obs(obs)
                self.mem.put_effect(a, r_normalized, done)
                obs = obs_next
                score_tmp += float(r)

            score.append(score_tmp)
            score_max = max(score_max, score[-1])
            print('Episode: {0} \n timestep {1} \n score {2:6f} \n best score {3:6f}'
                  .format(epoch, cnt_time_step, score[-1], score_max))

    def __get_q_batch(self, phi_batch, a_batch):
        q_value = self.net(phi_batch)
        idx = a_batch.view(self.args.size_minibatch, -1)
        q_value = torch.gather(q_value, 1, idx)
        q_value = q_value.view(self.args.size_minibatch)

        return q_value

    def __action_selection(self, phi):
        if self.args.command == "train" and random.uniform(0, 1) < self.epsilon:
            action = random.sample(self.action_space, 1)[0]
            self.epsilon = max(
                self.epsilon - (self.args.epsilon_init - self.args.epsilon_final) / self.args.epsilon_final_frame,
                self.args.epsilon_final)
        else:
            with torch.no_grad():
                q_value = self.net(torch.from_numpy(phi).unsqueeze(0).type(self.dtype)).cpu()
            q_value = q_value.view(-1)
            _, idx = q_value.max(dim=0)
            idx = np.asscalar(idx.cpu().numpy())
            action = self.action_space[idx]

        return action

    def __get_loss(self, phi_batch, a_batch, r_batch, phi_next_batch, done_batch):
        q_target = self.net_target(phi_next_batch).detach()
        q_target_max_a, _ = q_target.max(dim=1)
        y = r_batch + self.args.gamma * (1 - done_batch) * q_target_max_a
        q_value = self.__get_q_batch(phi_batch, a_batch)
        loss = pow(y - q_value, 2)

        return loss

    def __save_statistic(self, label, nums, save_path=None):
        n = np.arange(len(nums[1]))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for i in range(1, 3):
            filename_fig = os.path.join(save_path,
                                        "{}_{}_{}_{}.png".format(
                                            label[i],
                                            self.args.game,
                                            self.args.model_type,
                                            self.args.start_time))
            plt.figure()
            plt.plot(n, nums[i])
            plt.ylabel(label[i])
            plt.xlabel(label[0])
            plt.savefig(filename_fig)
            plt.close()

        filename_csv = os.path.join(save_path,
                                    "{}_{}_{}.csv".format(
                                        self.args.game,
                                        self.args.model_type,
                                        self.args.start_time))
        if len(nums[1]) == 1:
            with open(filename_csv, 'w+') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(label)
        with open(filename_csv, 'a+') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([nums[0], nums[1][-1], nums[2][-1]])


class Agent_DDPG(Agent):
    def __init__(self, args, game):
        super(Agent_DDPG, self).__init__(args, game)

        self.actor = model.DDPG_Actor(np.prod(game.dim_obs),
                                      np.prod(game.dim_action),
                                      torch.from_numpy(game.action_interval[1])
                                      .type(self.dtype)).type(self.dtype)
        self.actor_target = model.DDPG_Actor(np.prod(game.dim_obs),
                                             np.prod(game.dim_action),
                                             torch.from_numpy(game.action_interval[1])
                                             .type(self.dtype)).type(self.dtype)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = model.DDPG_Critic(np.prod(game.dim_obs),
                                        np.prod(game.dim_action)).type(self.dtype)
        self.critic_target = model.DDPG_Critic(np.prod(game.dim_obs),
                                               np.prod(game.dim_action)).type(self.dtype)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.mem = memory.MemoryReplay(args.size_replay_memory,
                                       game.dim_obs,
                                       args.len_history_frame,
                                       action_dim=self.game.env.action_space.shape,
                                       is_image=False)

        self.action_interval = game.action_interval
        self.cnt_action_repeat = 0

        if args.command == "train":
            if self.args.optim_method == "rmsprop":
                self.opt_actor = optim.RMSprop(self.actor.parameters(),
                                               lr=self.args.lr_actor,
                                               alpha=self.args.rmsprop_alpha,
                                               eps=self.args.rmsprop_epsilon)
                self.opt_critic = optim.RMSprop(self.critic.parameters(),
                                               lr=self.args.lr_critic,
                                               alpha=self.args.rmsprop_alpha,
                                               eps=self.args.rmsprop_epsilon)
            elif self.args.optim_method == "adam":
                self.opt_actor = optim.Adam(self.actor.parameters(),
                                            lr=self.args.lr_actor)
                self.opt_critic = optim.Adam(self.critic.parameters(),
                                             lr=self.args.lr_critic,
                                             weight_decay=self.args.weight_decay_critic)
            # self.epsilon = args.epsilon_init
            self.cnt_model_update = 0
            self.cnt_iter = 0
            self.save_path_results = os.path.join(args.save_path, "results")
            self.save_path_logs = os.path.join(args.save_path, "logs")
            self.save_path_models = os.path.join(args.save_path, "models")
        elif args.command == "play":
            self.load_path_models = os.path.join(args.load_path, "models")

        self.writer = SummaryWriter()

    def memory_fill(self):
        print("==========Initialize the replay memory===========")

        obs = self.game.env.reset()

        self.mem.put_obs(obs)
        for _ in range(int(self.args.size_replay_start/self.args.num_action_repeat)):
            a = self.__action_selection(obs)
            while self.cnt_action_repeat < self.args.num_action_repeat:
                obs_next, r, done, info = self.game.env.step(a)

                self.mem.put_effect(a, r, done)
                if done:
                    obs = self.game.env.reset()
                else:
                    obs = obs_next
                self.mem.put_obs(obs)
                self.cnt_action_repeat += 1
            self.cnt_action_repeat = 0

    def train_game(self):
        print("==========Start training===========")

        score = []
        score_mean = []
        score_mean_max = -float('inf')
        q_value = []
        cnt_time_step = 0

        # Initiate logger
        logger = logging.getLogger()
        if not os.path.exists(self.save_path_logs):
            os.makedirs(self.save_path_logs)
        file_name_logger = os.path.join(self.save_path_logs,
                                        "train_info_{}_{}_{}_{}.log".format(
                                            self.args.game,
                                            self.args.model_type,
                                            self.args.optim_method,
                                            self.args.start_time))
        if os.path.exists(file_name_logger):
            os.remove(file_name_logger)
        handler_file = logging.FileHandler(file_name_logger)
        handler_stream = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: \n %(message)s')
        handler_file.setFormatter(formatter)
        handler_stream.setFormatter(formatter)
        logger.addHandler(handler_file)
        logger.addHandler(handler_stream)
        logger.setLevel(logging.INFO)

        for epoch in range(self.args.num_epoch):
            obs = self.game.env.reset()

            self.mem.put_obs(obs)
            done = False
            score_tmp = 0
            q_tmp = 0
            a = 0
            while not done:
                cnt_time_step += 1

                # Select a new action every self.num_action_repeat frames
                if self.cnt_action_repeat == 0:
                    a = self.__action_selection(obs)
                    self.cnt_model_update = (self.cnt_model_update + 1) % self.args.freq_update
                self.cnt_action_repeat = (self.cnt_action_repeat + 1) % self.args.num_action_repeat

                # Execute the action
                obs_next, r, done, info = self.game.env.step(a)
                obs_next = obs_next.squeeze()

                self.mem.put_effect(a, r, done)
                obs = obs_next
                self.mem.put_obs(obs)
                score_tmp += float(r)

                # Update Q network every self.freq_update times of action selections
                if self.cnt_model_update == 0:
                    # Sample from replay memory
                    phi_b, a_b, r_b, phi_next_b, done_b = self.mem.sample(self.args.size_minibatch)
                    phi_batch = torch.from_numpy(phi_b).type(self.dtype)
                    a_batch = torch.from_numpy(a_b.reshape(self.args.size_minibatch, -1)).type(self.dtype)
                    r_batch = torch.from_numpy(r_b).reshape(self.args.size_minibatch, -1).type(self.dtype)
                    phi_next_batch = torch.from_numpy(phi_next_b).type(self.dtype)
                    done_batch = torch.from_numpy(done_b.astype(float)
                                                  .reshape(self.args.size_minibatch, -1)).type(self.dtype)

                    # Compute loss function
                    loss_critic, loss_actor = self.__get_loss(phi_batch, a_batch, r_batch, phi_next_batch, done_batch)

                    # Update actor and critic networks
                    self.opt_critic.zero_grad()
                    loss_critic.backward()
                    self.opt_critic.step()

                    self.opt_actor.zero_grad()
                    loss_actor.backward()
                    self.opt_actor.step()

                    # Update target actor and target critic networks
                    for param, param_target in zip(self.critic.parameters(), self.critic_target.parameters()):
                        param_target.data.copy_(self.args.tau * param + (1 - self.args.tau) * param_target)

                    for param, param_target in zip(self.actor.parameters(), self.actor_target.parameters()):
                        param_target.data.copy_(self.args.tau * param + (1 - self.args.tau) * param_target)

                    self.cnt_iter += 1

                if cnt_time_step % self.args.freq_log == 0:
                    self.writer.add_graph(self.critic, (phi_batch, a_batch))
                    self.writer.add_graph(self.actor, phi_batch)
                    self.writer.add_scalar('critic_loss', loss_critic, cnt_time_step)
                    self.writer.add_scalar('actor_loss', loss_actor, cnt_time_step)
                    if len(score) > 0:
                        self.writer.add_scalar('reward', score[-1])

            score.append(score_tmp)
            score_mean.append(np.mean(score[-100:]))
            if score_mean[-1] > score_mean_max and epoch > 1000:
                if not os.path.exists(self.save_path_models):
                    os.makedirs((self.save_path_models))
                file_name_model = os.path.join(self.save_path_models,
                                               "{}_{}_{}_{}.pt".format(
                                                   self.args.game,
                                                   self.args.model_type,
                                                   self.args.optim_method,
                                                   self.args.start_time))
                torch.save({"critic": self.critic.state_dict(),
                            "actor": self.actor.state_dict()},
                           file_name_model)
            if epoch > 1000:
                score_mean_max = max(score_mean_max, score_mean[-1])
            q_value.append(q_tmp)
            self.__save_statistic(['Training_epoch', 'Average_score', 'Average_action_value'],
                                   [epoch, score_mean, q_value], self.save_path_results)
            logging.info('Episode: {0} \n timestep {1} \n mean score {2:6f} \n best mean score {3:6f}'
                  .format(epoch, cnt_time_step, score_mean[-1], score_mean_max))

    def play_game(self):
        print("==========Start playing===========")
        file_name_model = os.path.join(self.load_path_models,
                                       "{}_{}_{}_{}.pt".format(
                                           self.args.game,
                                           self.args.model_type,
                                           self.args.optim_method,
                                           self.args.start_time))
        print(file_name_model)
        try:
            self.critic.load_state_dict(torch.load(file_name_model)["critic"])
            self.actor.load_state_dict(torch.load(file_name_model)["actor"])
        except FileNotFoundError:
            print('File does not exist!')

        score = []
        score_max = -float('inf')
        cnt_time_step = 0

        for epoch in range(self.args.num_epoch):
            obs = self.game.env.reset()
            for _ in range(self.args.len_history_frame * self.args.num_action_repeat):
                a = self.game.env.action_space.sample()
                while self.cnt_action_repeat < self.args.num_action_repeat:
                    obs_next, r, done, info = self.game.env.step(a)
                    self.mem.put_obs(obs)
                    self.mem.put_effect(a, r, done)
                    if done:
                        obs = self.game.env.reset()
                    else:
                        obs = obs_next
                    self.cnt_action_repeat += 1
                self.cnt_action_repeat = 0

            done = False
            score_tmp = 0
            a = 0
            while not done:
                cnt_time_step += 1

                # Select a new action every self.num_action_repeat frames
                if self.cnt_action_repeat == 0:
                    self.mem.put_obs(obs)
                    phi = self.mem.encode_frame()
                    a = self.__action_selection(phi)
                self.cnt_action_repeat = (self.cnt_action_repeat + 1) % self.args.num_action_repeat

                # Execute the action
                self.game.env.render()
                time.sleep(0.01)
                obs_next, r, done, info = self.game.env.step(a)

                # self.mem.put_obs(obs)
                self.mem.put_effect(a, r, done)
                obs = obs_next
                score_tmp += float(r)

            score.append(score_tmp)
            score_max = max(score_max, score[-1])
            print('Episode: {0} \n timestep {1} \n score {2:6f} \n best score {3:6f}'
                  .format(epoch, cnt_time_step, score[-1], score_max))

    def __action_selection(self, phi):
        if len(phi.shape) == len(self.game.env.observation_space.shape):
            phi = np.expand_dims(phi, axis=0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(torch.from_numpy(phi).type(self.dtype)).cpu().numpy()
        if self.args.command == "train":
            action += random.normalvariate(0, self.args.exploration_noise)
            action.clip(self.game.action_interval[0], self.game.action_interval[1])

        return action.squeeze(0)

    def __get_loss(self, phi_batch, a_batch, r_batch, phi_next_batch, done_batch):
        self.critic.train()
        self.critic_target.train()
        self.actor.train()
        self.actor_target.train()
        q_target = self.critic_target(phi_next_batch, self.actor_target(phi_next_batch)).detach()
        y = r_batch + self.args.gamma * (1 - done_batch) * q_target
        q = self.critic(phi_batch, a_batch)
        loss_critic = F.mse_loss(y, q)

        loss_actor = -self.critic(phi_batch, self.actor(phi_batch)).mean()

        return loss_critic, loss_actor

    def __save_statistic(self, label, nums, save_path=None):
        n = np.arange(len(nums[1]))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for i in range(1, 3):
            filename_fig = os.path.join(save_path,
                                        "{}_{}_{}_{}.png".format(
                                            label[i],
                                            self.args.game,
                                            self.args.model_type,
                                            self.args.start_time))
            plt.figure()
            plt.plot(n, nums[i])
            plt.ylabel(label[i])
            plt.xlabel(label[0])
            plt.savefig(filename_fig)
            plt.close()

        filename_csv = os.path.join(save_path,
                                    "{}_{}_{}.csv".format(
                                        self.args.game,
                                        self.args.model_type,
                                        self.args.start_time))
        if len(nums[1]) == 1:
            with open(filename_csv, 'w+') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(label)
        with open(filename_csv, 'a+') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([nums[0], nums[1][-1], nums[2][-1]])