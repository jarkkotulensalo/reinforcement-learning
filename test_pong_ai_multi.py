"""
This is an example on how to use the two player Wimblepong environment
with two SimpleAIs playing against each other
"""

import gym
import argparse
import matplotlib.pyplot as plt
import wimblepong
import agent_jack
import yaml

from gym import logger as gymlogger
gymlogger.set_level(40) #error onlyfrom gym import logger as gymlogger

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
parser.add_argument('--config', default='config.yaml')
args = parser.parse_args()

# Remove registry
"""
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'WimblepongVisualMultiplayer-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
"""
# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
# Number of episodes/games to play


# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = wimblepong.SimpleAi(env, opponent_id)

# LOADPATH = "pretrained_models/weights_Jack-v2_1200000.mdl"
config = yaml.load(open(args.config))
load_path = config['path_pretrained_model']
agent_config = config['agent_params']
optim_config = agent_config['optim_params']
player = agent_jack.Agent(env, player_id, load_path=load_path, optim_params=optim_config)
print(player.get_name())

# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())

episodes = 20

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
