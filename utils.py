from collections import namedtuple
import random
import pickle
import numpy as np




Transition = namedtuple('Transition',
                        ('ob1', 'action1', 'next_ob1', 'rew1', 'done'))

def rgb2gray(rgb):
    return np.dot(rgb[:,:,:3], [0.2989, 0.5870, 0.1140])

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

    def save_memory(self,savepath):
        with open(savepath,'wb') as f:
            pickle.dump(self.memory,f)

    def load_memory(self, loadpath, loadpath2=None):
        with open(loadpath,'rb') as f:
            transitions = pickle.load(f)
        if loadpath2 != None:
            with open(loadpath2,'rb') as f:
                transitions2 = pickle.load(f)
        return transitions+transitions2
