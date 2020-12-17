import numpy as np
import random
import torch


class MemoryReplay(object):
    def __init__(self, size_memory, state_dim, len_history_frame, action_dim=(1,), num_game=1, is_image=True):
        super(MemoryReplay, self).__init__()
        self.size_memory = size_memory
        self.len_history_frame = len_history_frame
        self.num_game = num_game
        self.state_dim = state_dim
        # self.phi stores len_history_frame+1 frames, in which :len_history_frame+1 is phi_t, and 1: is phi_t+1
        self.phi = np.zeros((num_game, size_memory) + state_dim, dtype=(np.uint8 if is_image else np.float32))
        self.action = np.zeros((num_game, size_memory)+action_dim, dtype=np.int32)
        self.reward = np.zeros((num_game, size_memory), dtype=np.float32)
        self.done = np.array([[False for _ in range(size_memory)] for _ in range(num_game)])
        self.count = 0
        self.full = False
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    def put_obs(self, obs):
        self.phi[:, self.count] = obs

    def put_effect(self, action, reward, done):
        self.action[:, self.count] = action
        self.reward[:, self.count] = reward
        self.done[:, self.count] = done
        if (not self.full) and (self.count == self.size_memory - 1):
            self.full = True
        self.count = (self.count + 1) % self.size_memory

    def sample(self, size_batch):
        if self.full:
            idx = random.sample(list(range(self.size_memory-1)), size_batch)
        else:
            idx = random.sample(list(range(self.count)), size_batch)
        phi = np.concatenate([self.encode_frame(i)[None] for i in idx], 0)
        action = self.action[0, idx]
        reward = self.reward[0, idx]
        phi_next = np.concatenate([self.encode_frame(i+1)[None] for i in idx], 0)
        done = self.done[0, idx]
        return phi, action, reward, phi_next, done

    def encode_frame(self, count=None):
        idx_end = count + 1 if count else self.count + 1
        idx_start = [idx_end - self.len_history_frame
                     if idx_end >= self.len_history_frame else 0
                     for _ in range(self.num_game)]
        num_missing = [0 for _ in range(self.num_game)]
        frame_encode = np.zeros_like(self.phi[:, 0:self.len_history_frame])
        for n in range(self.num_game):
            for i in range(idx_start[n], idx_end):
                if self.done[n, i]:
                    idx_start[n] = i + 1
            num_missing[n] = self.len_history_frame - (idx_end - idx_start[n])
            for i in range(num_missing[n], self.len_history_frame):
                try:
                    frame_encode[n, i] = self.phi[0, idx_start[n]+i-num_missing[n]]
                except IndexError:
                    print('Index error. idx_start = {}, idx_end = {}, i = {}, phi.shape = {}, frame_encode.shape = '
                          .format(idx_start, idx_end, i, self.phi.shape, frame_encode.shape))
                    exit(1)
        return frame_encode.squeeze()

    def get_return(self, net, obs_next, gamma):
        phi_next = torch.zeros((self.num_game, self.len_history_frame) + self.state_dim).type(self.dtype)
        phi_next[:, :self.len_history_frame - 1] = torch.from_numpy(self.encode_frame()[:, 1:]).type(self.dtype)
        phi_next[:, -1] = torch.from_numpy(obs_next).type(self.dtype)
        ret = torch.zeros((self.num_game, self.size_memory+1)).type(self.dtype)
        with torch.no_grad():
            ret[:, -1] = net(phi_next/255.0)[2].squeeze()
        for i in range(self.size_memory - 1, -1, -1):
            ret[:, i] = torch.from_numpy(self.reward[:, i]).type(self.dtype)\
                        + gamma * torch.from_numpy(self.done[:, i]).type(self.dtype) * ret[:, i+1]
        return ret

    def get_value_prob_entropy(self, net, dist_fn):
        phi = torch.zeros(
            (self.num_game, self.size_memory, self.len_history_frame) + self.state_dim).type(self.dtype)
        for ns in range(self.size_memory):
            phi[:, ns] = torch.from_numpy(self.encode_frame(count=ns)).type(self.dtype)
        prob, log_prob, value = net(phi.view((-1, self.len_history_frame) + self.state_dim)/255.0)
        dist = dist_fn(prob)
        action = torch.from_numpy(self.action).type(self.dtype).view(-1)
        log_prob_action = dist.log_prob(action)
        entropy = dist.entropy()
        return value.view(self.num_game, self.size_memory), \
               log_prob_action.view(self.num_game, self.size_memory),\
               entropy.view(self.num_game, self.size_memory)
