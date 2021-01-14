
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity, dagger_files=None, frame_stacks=2):
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
            processed_state = self._preprocess(transitions[0].state.cpu().data.numpy())
            for i, transition in enumerate(transitions):
                #if i % 3 == 0:  # 2-5 acc. to random frame skipping in env
                    # print(f"(transition.state.shape {transition.state.shape}")
                processed_next_state = self._preprocess(transition.next_state.cpu().data.numpy())
                self.push(processed_state,
                          transition.action,
                          processed_next_state,
                          transition.reward,
                          transition.done)

                processed_state = processed_next_state
                # print(f"processed_state {processed_state.shape}")

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

def plot_rewards(episode_num, rewards_avg_list, frames_avg_list):
    """
    Plot rewards and avg frames per episode.
    :param episode_num:
    :param rewards_avg_list:
    :param frames_avg_list:
    :return:
    """

    if (episode_num % 10000 == 0 and episode_num > 0) or (episode_num == 1000):
        fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, figsize=(6.4, 4.8 * 2))
        x = np.arange(len(rewards_avg_list))
        x = x * 200
        ax1.plot(x, rewards_avg_list)
        ax1.set_xlabel(f"Number of episodes")
        ax1.set_ylabel(f"Avg. reward for 200 episodes")

        ax2.plot(x, frames_avg_list)
        ax2.set_xlabel(f"Number of episodes")
        ax2.set_ylabel(f"Avg. frame duration for 200 episodes")

        fig.tight_layout()
        plt.savefig(f"./plots/rewards_{episode_num}.png")
        print(f"Learning plot saved after episode {episode_num}.")

def plot_exploration_strategy(num_episodes, EXP_EPISODES, glie_a, exp_end):
    x = np.arange(num_episodes)
    y = np.zeros(num_episodes)
    for episode_num in range(0, num_episodes):
        if episode_num < EXP_EPISODES:
            y[episode_num] = glie_a / (glie_a + episode_num)
        else:
            y[episode_num] = exp_end

    plt.ylabel('Exploration')
    plt.xlabel('Number of episodes')
    plt.plot(x, y)
    plt.savefig(f"./plots/exploration_{num_episodes}.png")



def calc_glie(episode_num, EXP_EPISODES, glie_a, exp_end):

    if episode_num < EXP_EPISODES - 1:
        eps = glie_a / (glie_a + episode_num)
    else:
        eps = exp_end
    return eps

def get_dagger_files(use_dagger):

    if use_dagger:
        dagger_files = ['./dagger/mem9-1.pickle',
                        './dagger/mem7-3.pickle',
                        './dagger/mem6-4.pickle',
                        './dagger/mem6-5.pickle',
                        './dagger/mem25-6.pickle']
    else:
        dagger_files = None
    return dagger_files