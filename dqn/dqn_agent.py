"""
Based on PyTorch DQN tutorial by Adam Paszke <https://github.com/apaszke>

BSD 3-Clause License

Copyright (c) 2017, Pytorch contributors
All rights reserved.
"""

import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
# from dqn.utils import Transition, ReplayMemory

from wimblepong import Wimblepong
import torchvision.transforms as transforms


class DQN(nn.Module):
    def __init__(self, action_space_dim, hidden=32, batch_size=64):
        super(DQN, self).__init__()
        #self.hidden = hidden
        #self.fc1 = nn.Linear(6400, hidden)
        #self.fc2 = nn.Linear(hidden, action_space_dim)

        self.action_space = action_space_dim
        self.hidden = hidden
        self.batch_size = batch_size
        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=32,
                                     kernel_size=3,
                                     stride=2)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 2)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, 2)
        self.reshaped_size = 128 * 11 * 11
        # self.reshaped_size = 90112

        self.fc1 = torch.nn.Linear(self.reshaped_size, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, action_space_dim)


    def forward(self, x):
        x = self.conv1(x)
        # print(f"forward x {x.shape}")
        x = F.relu(x)
        x = self.conv2(x)
        # print(f"forward x {x.shape}")
        x = F.relu(x)
        x = self.conv3(x)
        # print(f"forward x {x.shape}")
        x = F.relu(x)

        x = x.reshape(x.shape[0], self.reshaped_size)
        # print(f"forward x {x.shape}")
        x = self.fc1(x)
        # print(f"forward x {x.shape}")
        x = F.relu(x)
        x = self.fc2(x)
        # print(f"forward x {x.shape}")
        return x

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent(object):
    def __init__(self, env, player_id, n_actions, replay_buffer_size=50000,
                 batch_size=32, hidden_size=16, gamma=0.98):
        if type(env) is not Wimblepong:
            raise TypeError("I'm not a very smart AI. All I can play is Wimblepong.")
        self.env = env
        # Set the player id that determines on which side the ai is going to play
        self.player_id = player_id
        # Ball prediction error, introduce noise such that SimpleAI reflects not
        # only in straight lines
        self.bpe = 4
        self.name = "kingfisher"

        self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training with {self.train_device}")
        # self.train_device = torch.device("cpu")
        self.prev_obs = None

        self.n_actions = n_actions
        self.policy_net = DQN(n_actions, hidden_size, batch_size).to(self.train_device)
        self.target_net = DQN(n_actions, hidden_size, batch_size).to(self.train_device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=1e-3)
        self.memory = ReplayMemory(replay_buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma

    def update_network(self, updates=1):
        for _ in range(updates):
            self._do_network_update()

    # noinspection PyTypeChecker
    def _do_network_update(self):
        if len(self.memory) < self.batch_size:
            return

        #print(f"len(self.memory) is {len(self.memory)}")
        #print(f"self.batch_size is {self.batch_size}")

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # noinspection PyTypeChecker
        non_final_mask = 1-torch.tensor(batch.done, dtype=torch.uint8)
        non_final_next_states = [s for nonfinal,s in zip(non_final_mask,
                                     batch.next_state) if nonfinal > 0]
        non_final_next_states = torch.stack(non_final_next_states).to(self.train_device)
        state_batch = torch.stack(batch.state).to(self.train_device)
        action_batch = torch.cat(batch.action).to(self.train_device)
        reward_batch = torch.cat(batch.reward).to(self.train_device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # print("state_batch")
        # print(state_batch.shape)
        # print("action_batch")
        # print(action_batch.shape)
        # print(action_batch)
        # print(f"self.policy_net(state_batch) {self.policy_net(state_batch).shape}")
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        # print(f"state_action_values {state_action_values.shape}")
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size).to(self.train_device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Task 4: TODO: Compute the expected Q values
        expected_state_action_values = reward_batch + self.gamma * next_state_values

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(),
                                expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1e-1, 1e-1)
        self.optimizer.step()

    def get_action(self, observation, epsilon=0.00):
        state = self.preprocess(observation)
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                state = torch.from_numpy(state).float().unsqueeze(1).to(self.train_device)
                # print(f"action state is {state.shape}")
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()
        else:
            return random.randrange(self.n_actions)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, observation, action, next_obs, reward, done):
        state = self.preprocess(observation)
        next_state = self.preprocess(next_obs)

        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = torch.from_numpy(next_state).float()
        state = torch.from_numpy(state).float()
        self.memory.push(state, action, next_state, reward, done)

    def get_name(self):
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def load_model(self):
        """
        state_dict = torch.load(args.test)
        :return:
        """
        weights = torch.load("model.mdl")
        self.target_net.load_state_dict(weights, strict=False)
        return

    def reset(self):
        # Nothing to done for now...
        return

    def preprocess(self, observation):
        #print(f"observation {observation.shape}")
        observation = observation[::2, ::2].mean(axis=-1)
        #print(f"observation {observation.shape}")
        observation = np.expand_dims(observation, axis=0)
        #print(f"observation {observation.shape}")
        observation[observation == 43] = 0 # erase background (background type 1)
        #print(f"observation {observation.shape}")
        observation[observation != 0] = 1  # everything else (paddles, ball) just set to 1
        #print(f"observation {observation.shape}")

        # observation = np.expand_dims(observation, axis=0)
        # observation = observation.astype(np.float).ravel()
        #(f"observation {observation.shape}")

        if self.prev_obs is None:
            self.prev_obs = observation
            stack_ob = np.zeros(100*100)
        else:
            stack_ob = observation - self.prev_obs
        """
        if self.prev_obs is None:
            self.prev_obs = observation
        stack_ob = np.concatenate((self.prev_obs, observation), axis=-1)
        #print(f"stack_ob {stack_ob.shape}")
        stack_ob = torch.from_numpy(stack_ob).float().unsqueeze(0)
        #print(f"stack_ob {stack_ob.shape}")
        stack_ob = stack_ob.transpose(1, 3)
        print(f"stack_ob {stack_ob.shape}")
        self.prev_obs = observation
        """
        # print(f"stack_ob {stack_ob.shape}")
        return stack_ob

    def preprocess_observations(self, observation, input_dimensions=80*80):
        """ convert the 210x160x3 uint8 frame into a 6400 float vector """
        processed_observation = observation[35:195]  # crop
        processed_observation = downsample(processed_observation)
        processed_observation = remove_color(processed_observation)
        processed_observation = remove_background(processed_observation)
        processed_observation[processed_observation != 0] = 1  # everything else (paddles, ball) just set to 1
        # Convert from 80 x 80 matrix to 1600 x 1 matrix
        processed_observation = processed_observation.astype(np.float).ravel()

        # subtract the previous frame from the current one so we are only processing on changes in the game
        if self.prev_obs is not None:
            observation = processed_observation - self.prev_obs
        else:
            observation = np.zeros(input_dimensions)
        # store the previous frame so we can subtract from it next time
        prev_processed_observations = processed_observation
        return observation, prev_processed_observations

def downsample(image):
    # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
    return image[::2, ::2, :]

def remove_color(image):
    """Convert all color (RGB is the third dimension in the image)"""
    return image[:, :, 0]

def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image

def prepro(self, I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 43] = 0  # erase background (background type 1)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()