
from collections import namedtuple
import numpy as np
import pickle
import random


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity, dagger_files=None, frame_stacks=4):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.dagger_files = dagger_files
        self.frame_stacks = frame_stacks
        self.prev_obs = None
        self._load_memory()

    def __len__(self):
        return len(self.memory)

    def _load_memory(self):
        transitions = []
        if self.dagger_files != None:
            for fpath in self.dagger_files:
                with open(fpath, 'rb') as f:
                    transitions += pickle.load(f)
            print(f"Total dagger input memory frames {len(transitions)}")

            transitions_processed = []
            total_i = 0
            for i, transition in enumerate(transitions):
                if i % self.frame_stacks == 0:
                    # print(f"(transition.state.shape {transition.state.shape}")
                    processed_state = self._preprocess(transition.state.cpu().data.numpy())
                    processed_next_state = self._preprocess_next(transition.next_state.cpu().data.numpy())
                    # print(f"processed_state {processed_state.shape}")

                    self.push(processed_state,
                                transition.action,
                                processed_next_state,
                                transition.reward,
                                transition.done)
                    total_i += 1
            print(f"Total dagger output memory frames {total_i}")
        return

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def save_memory(self, savepath):
        with open(savepath, 'wb') as f:
            pickle.dump(self.memory, f)

    def _preprocess(self, observation):
        # print(f"observation {observation.shape}")
        # observation = observation[::2, ::2].mean(axis=-1)

        # observation = np.dot(observation, [0.2989, 0.5870, 0.1140]).astype(np.uint8)  # convert to greyscale
        # print(f"observation {observation.shape}")
        observation = observation[::2, ::2]
        # print(f"observation {observation.shape}")
        observation = np.expand_dims(observation, axis=0)
        # print(f"observation {observation.shape}")

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

        self.prev_obs = stack_ob[1:self.frame_stacks, :, :]
        # print(f"self.prev_obs.shape[0] {self.prev_obs.shape[0]}")

        # print(f"stack_ob {stack_ob.shape}")
        return stack_ob

    def _preprocess_next(self, observation):

        observation = observation[::2, ::2]
        # print(f"observation {observation.shape}")
        observation = np.expand_dims(observation, axis=0)
        # print(f"observation {observation.shape}")

        #print(f"self.prev_obs {self.prev_obs.shape}")
        stack_ob = np.concatenate((self.prev_obs, observation), axis=0)
        # print(f"stack_ob {stack_ob.shape}")
        return stack_ob

    def _stack_frames(self, stack_ob, obs):
        """
        Stack a sequence of frames into one array to until 4 frames is stacked
        :param stack_ob:
        :param obs:
        :return:
        """
        return np.concatenate((stack_ob, obs), axis=0)