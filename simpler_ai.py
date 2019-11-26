from wimblepong import Wimblepong
from utils import ReplayMemory, Transition
import time
import random
from scratch import getkey
import torch


class NOTAI(object):
    def __init__(self, env, player_id=1):
        if type(env) is not Wimblepong:
            raise TypeError("I'm not a very smart AI. All I can play is Wimblepong.")
        self.env = env
        # Set the player id that determines on which side the ai is going to play
        self.player_id = player_id
        # Ball prediction error, introduce noise such that SimpleAI reflects not
        # only in straight lines
        self.bpe = 4
        self.name = "NOTAI"
        self.memory = ReplayMemory(1000000)

    def get_name(self):
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def get_action(self, ob=None):
        """
        Interface function that returns the action that the agent took based
        on the observation ob
        """

        action = int(getkey())
        print(action)


        return action

    def reset(self):
        # Nothing to done for now...
        return
    def store_transition(self, ob, action, next_ob, reward, done):
        action = torch.Tensor([action]).long()
        reward = torch.tensor([reward], dtype=torch.float32)
        next_ob = torch.from_numpy(next_ob).float()
        ob = torch.from_numpy(ob).float()
        self.memory.push(ob, action, next_ob, reward, done)
