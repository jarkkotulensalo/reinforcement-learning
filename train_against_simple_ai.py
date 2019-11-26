"""
This is an example on how to use the two player Wimblepong environment
with two SimpleAIs playing against each other
"""
import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import numpy as np
import argparse
import wimblepong
from dqn_agent import DQN_agent
import torch
from utils import rgb2gray

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
# Number of episodes/games to play
episodes = 100000

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = wimblepong.SimpleAi(env, opponent_id)
#player = wimblepong.SimpleAi(env, player_id)
player = DQN_agent(env, player_id)
print(player.get_name())
# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())

#GLIE
glie_a = 50
TARGET_UPDATE = 5


win1 = 0
for i in range(0,episodes):
    done = False
    if i/episodes < 0.5:
        eps = glie_a/(glie_a+i*2)
    else:
        eps = 0
    ob1, ob2 = env.reset()
    while not done:
        # Get the actions from both SimpleAIs

        action1 = player.get_action(ob1,eps)
        action2 = opponent.get_action()
        # Step the environment and get the rewards and new observations
        (next_ob1, next_ob2), (rew1, rew2), done, info = env.step((action1, action2))
        #obi 200,200,3
        player.store_transition(ob1, action1, next_ob1, rew1, done)
        player.update_network()
        ob1 = next_ob1
        # Count the wins
        if rew1 == 10:
            win1 += 1
        if not args.headless:
            env.render()
        if done:
            observation= env.reset()
            print("episode {} over. Broken WR: {:.3f}".format(i, win1/(i+1)))
    if i % TARGET_UPDATE == 0:
        print("updating target_net")
        player.update_target_network()

    if i % 5000 == 0:
        torch.save(player.policy_net.state_dict(),
        "weights_%s_%d.mdl" % ("WimblepongVisualMultiplayer-v0", i))
