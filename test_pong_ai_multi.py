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
from PIL import Image
import agent_jack

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
episodes = 20
LOADPATH = "./weights_Jack-v0_360000.mdl"
# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = wimblepong.SimpleAi(env, opponent_id)
player = agent_jack.Agent(env, player_id)
print(player.get_name())
player.load_model(LOADPATH)
# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())

win1 = 0
for i in range(0, episodes):
    done = False
    obs1, obs2 = env.reset()
    while not done:
        # Get the actions from both SimpleAIs
        action1 = player.get_action(obs1)
        action2 = opponent.get_action()
        # Step the environment and get the rewards and new observations
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
        obs1 = ob1
        if rew1 == 10:
            win1 += 1
        if not args.headless:
            env.render()
        if done:
            print(f"episode {i} over. Wins {win1}/{i}")
