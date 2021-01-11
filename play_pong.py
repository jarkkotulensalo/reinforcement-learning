"""
This is an example on how to use the two player Wimblepong environment
with two SimpleAIs playing against each other
"""

import gym
import argparse
import matplotlib.pyplot as plt
import wimblepong
from agent import Agent
import yaml

from gym import logger as gymlogger
gymlogger.set_level(40) #error onlyfrom gym import logger as gymlogger

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
parser.add_argument('--config', default='config_play.yaml')
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = wimblepong.SimpleAi(env, opponent_id)

config = yaml.load(open(args.config), Loader=yaml.FullLoader)
episodes = config['num_episodes']
load_path = config['path_pretrained_model']
agent_config = config['agent_params']
optim_config = agent_config['optim_params']
player = Agent(env, player_id, load_path=load_path, optim_params=optim_config)
print(player.get_name())

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
            observation = env.reset()
            print(f"episode {i} over. Wins {win1}/{i}")
