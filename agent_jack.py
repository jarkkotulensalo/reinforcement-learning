"""
Based on PyTorch DQN tutorial by Adam Paszke <https://github.com/apaszke>

BSD 3-Clause License

Copyright (c) 2017, Pytorch contributors
All rights reserved.
"""

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from utils import Transition, ReplayMemory
from wimblepong import Wimblepong


class DQN(nn.Module):
    def __init__(self, action_space_dim, hidden=512, frame_stacks=4):
        super(DQN, self).__init__()
        self.action_space = action_space_dim
        self.hidden = hidden

        self.conv1 = torch.nn.Conv2d(in_channels=frame_stacks,
                                     out_channels=32,
                                     kernel_size=8,
                                     stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1)
        self.reshaped_size = 64 * 9 * 9
        self.fc1 = torch.nn.Linear(self.reshaped_size, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, action_space_dim)
        self._init_weights()

    def _init_weights(self):
        print(f"Initialisation of weights with xavier")
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                # print(f"init weights")
                torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print(f"forward x {x.shape}")
        x = F.relu(self.conv2(x))
        #print(f"forward x {x.shape}")
        x = F.relu(self.conv3(x))
        #print(f"forward x {x.shape}")
        x = x.reshape(x.shape[0], self.reshaped_size)
        #print(f"forward x {x.shape}")
        x = F.relu(self.fc1(x))
        # print(f"forward x {x.shape}")
        x = self.fc2(x)
        # print(f"forward x {x.shape}")
        return x


class Agent(object):
    def __init__(self, env, player_id, n_actions=3, replay_buffer_size=100000,
                 batch_size=32, hidden_size=512, gamma=0.99, lr=2.5e-4, save_memory=True,
                 frame_stacks=2, dagger_files=None, double_dqn=True):
        if type(env) is not Wimblepong:
            raise TypeError("I'm not a very smart AI. All I can play is Wimblepong.")

        print(f"testing with {frame_stacks} fram stacks")
        self.env = env
        # Set the player id that determines on which side the ai is going to play
        self.player_id = player_id
        # Ball prediction error, introduce noise such that SimpleAI reflects not
        # only in straight lines
        self.bpe = 4
        self.name = "jack_the_dagger"
        self.frame_stacks = frame_stacks

        self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training with {self.train_device}")
        # self.train_device = torch.device("cpu")
        self.prev_obs = None
        self.save_memory = save_memory

        self.batch_size = batch_size
        self.n_actions = n_actions
        self.policy_net = DQN(n_actions, hidden_size, frame_stacks).to(self.train_device)
        self.target_net = DQN(n_actions, hidden_size, frame_stacks).to(self.train_device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr, eps=1e-2, momentum=0.95)
        self.memory = ReplayMemory(replay_buffer_size, dagger_files, frame_stacks)
        self.batch_size = batch_size
        self.gamma = gamma
        self.double_dqn = double_dqn

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

        if self.save_memory:
            non_final_mask = 1-torch.tensor(batch.done, dtype=torch.uint8, device=self.train_device)
            non_final_next_states = [torch.tensor(s, dtype=torch.float, device=self.train_device) for nonfinal,s in
                                     zip(non_final_mask, batch.next_state) if nonfinal > 0]
            non_final_next_states = torch.stack(non_final_next_states).squeeze(1)
            state_batch = torch.tensor(batch.state, dtype=torch.float, device=self.train_device).squeeze(1)
            action_batch = torch.tensor(batch.action, device=self.train_device).unsqueeze(1)
            reward_batch = torch.tensor(batch.reward, dtype=torch.float, device=self.train_device)

        else:
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
        # print(f"self.policy_net(state_batch) {self.policy_net(state_batch).shape}")
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        # print(f"state_action_values {state_action_values.shape}")
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size).to(self.train_device)
        # Double DQN - Compute V(s_{t+1}) for all next states.
        if self.double_dqn:
            next_state_actions = self.policy_net(next_state_values).max(1, keepdim=True)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_state_actions)
        else:
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Task 4: TODO: Compute the expected Q values
        # next_state_values.volatile = False
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

    def _preprocess(self, observation):
        # print(f"observation {observation.shape}")
        # observation = observation[::2, ::2].mean(axis=-1)

        observation = np.dot(observation, [0.2989, 0.5870, 0.1140]).astype(np.uint8)  # convert to greyscale
        # print(f"observation {observation.shape}")
        observation = observation[::2, ::2]
        #print(f"observation {observation.shape}")
        observation = np.expand_dims(observation, axis=0)
        #print(f"observation {observation.shape}")

        if self.prev_obs is None:
            self.prev_obs = observation

        #print(f"self.prev_obs {self.prev_obs.shape}")
        stack_ob = np.concatenate((self.prev_obs, observation), axis=0)
        # print(f"stack_ob {stack_ob.shape}")

        # print(f"stack_ob.shape[0] {stack_ob.shape[0]}")
        while stack_ob.shape[0] < self.frame_stacks:
            stack_ob = self._stack_frames(stack_ob, observation)
            # print(f"stack_ob.shape[0] {stack_ob.shape[0]}")
        # print(f"stack_ob {stack_ob.shape}")
        # self.prev_obs = np.delete(stack_ob, (0), axis=0)

        self.prev_obs = stack_ob[1:self.frame_stacks, :, :]
        # print(f"self.prev_obs.shape[0] {self.prev_obs.shape[0]}")

        # print(f"stack_ob {stack_ob.shape}")
        return stack_ob

    def _stack_frames(self, stack_ob, obs):
        """
        Stack a sequence of frames into one array to until n frames is stacked
        :param stack_ob:
        :param obs:
        :return:
        """
        return np.concatenate((stack_ob, obs), axis=0)

    def get_action(self, observation, epsilon=0.3):
        # epsilon = 0.1
        sample = random.random()
        if sample > epsilon:
            state = self._preprocess(observation)
            with torch.no_grad():
                # print(f"state {state.shape}")
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.train_device)
                # print(f"state {state.shape}")
                q_values = self.policy_net(state)
                chosen_action = torch.argmax(q_values).item()
                # print(f"Chosen action is {chosen_action}")
                return chosen_action
        else:
            return random.randrange(self.n_actions)

    def get_name(self):
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def load_model(self, fpath):
        """
        state_dict = torch.load(args.test)
        :return:
        """
        weights = torch.load(fpath)
        self.policy_net.load_state_dict(weights, strict=False)
        print(f"Loaded model from {fpath}")
        return

    def reset(self):
        # Nothing to done for now...
        return

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, observation, action, next_obs, reward, done):
        state = self._preprocess(observation)
        next_state = self._preprocess(next_obs)
        # print(f"state.shape {state.shape}")

        if not self.save_memory:
            action = torch.Tensor([[action]]).long()
            reward = torch.tensor([reward], dtype=torch.float32)
            next_state = torch.from_numpy(next_state).float()
            state = torch.from_numpy(state).float()

        # print(f"state.shape {state.shape}")
        self.memory.push(state, action, next_state, reward, done)
