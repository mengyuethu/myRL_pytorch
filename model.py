import numpy as np
import torch
import torch.nn as nn


class Double_DQN(nn.Module):

    def __init__(self, input_img_size, input_channels, output_channels):
        super(Double_DQN, self).__init__()
        self.input_img_size = input_img_size
        self.in_features_linear = int(pow(((((input_img_size - 8)/4+1)-4)/2+1)-2, 2)*64)
        self.input_channels = np.array([input_channels, 32, 64, self.in_features_linear, 512])
        self.output_channels = np.array([32, 64, 64, 512, output_channels])

        self.conv1 = nn.Conv2d(self.input_channels[0], self.output_channels[0], kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(self.input_channels[1], self.output_channels[1], kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(self.input_channels[2], self.output_channels[2], kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.input_channels[3], self.output_channels[3])
        self.fc2 = nn.Linear(self.input_channels[4], self.output_channels[4])

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = x.view(-1, self.input_channels[3])
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Dueling_DQN(nn.Module):

    def __init__(self, input_img_size, input_channels, output_channels):
        super(Dueling_DQN, self).__init__()
        self.input_img_size = input_img_size
        self.in_features_linear = int(pow(((((input_img_size - 8)/4+1)-4)/2+1)-2, 2)*64)
        self.input_channels = np.array([input_channels, 32, 64, self.in_features_linear, 512])
        self.output_channels = np.array([32, 64, 64, 512, output_channels])

        self.conv1 = nn.Conv2d(self.input_channels[0], self.output_channels[0], kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(self.input_channels[1], self.output_channels[1], kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(self.input_channels[2], self.output_channels[2], kernel_size=3, stride=1)
        self.fc11 = nn.Linear(self.input_channels[3], self.output_channels[3])
        self.fc12 = nn.Linear(self.input_channels[3], self.output_channels[3])
        self.fc21 = nn.Linear(self.input_channels[4], self.output_channels[4])
        self.fc22 = nn.Linear(self.input_channels[4], 1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = x.view(-1, self.input_channels[3])
        x1 = nn.functional.relu(self.fc11(x))
        x2 = nn.functional.relu(self.fc12(x))
        x1 = self.fc21(x1)
        x2 = self.fc22(x2)
        x = x2 + x1 - x1.mean(dim=1).reshape(x2.shape)
        return x


class DDPG_Actor(nn.Module):

    def __init__(self, state_dim, action_dim, action_max):
        super(DDPG_Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.bn1 = nn.BatchNorm1d(state_dim)
        self.bn2 = nn.BatchNorm1d(400)
        self.bn3 = nn.BatchNorm1d(300)
        self.action_max = action_max

    def forward(self, s):
        x = nn.functional.relu(self.fc1(self.bn1(s)))
        x = nn.functional.relu(self.fc2(self.bn2(x)))
        x = self.action_max * torch.tanh(self.fc3(self.bn3(x)))
        return x


class DDPG_Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(DDPG_Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400 - action_dim)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)
        self.bn1 = nn.BatchNorm1d(state_dim)
        self.bn2 = nn.BatchNorm1d(400 - action_dim)

    def forward(self, s, a):
        x = nn.functional.relu(self.fc1(self.bn1(s)))
        x = nn.functional.relu(self.fc2(torch.cat([self.bn2(x), a], dim=1)))
        x = self.fc3(x)
        return x
