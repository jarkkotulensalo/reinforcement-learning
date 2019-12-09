"""
This is an example on how to use the two player Wimblepong environment with one
agent and the SimpleAI
"""
import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import numpy as np
import torch
import argparse
import warnings
import wimblepong
import agent_jack

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--housekeeping", action="store_true", help="Plot, player and ball positions and velocities at the end of each episode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualSimpleAI-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps

# Number of episodes/games to play
episodes = 100000  # 100000
n_actions = 3
replay_buffer_size = 100000
batch_size = 32
hidden_size = 512
gamma = 0.99
lr = 2.5e-4
frame_stacks = 2
EXP_EPISODES = 50000
glie_a = round(0.1 / 0.9 * EXP_EPISODES, 0)

# https://towardsdatascience.com/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756
TARGET_UPDATE_FRAMES = 2500
dagger_files = ['./mem9-1.pickle',
                './mem7-3.pickle',
                './mem6-4.pickle',
                './mem6-5.pickle',
                './mem25-6.pickle']

# dagger_files = None
# Define the player
player_id = 1
# Set up the player here. We used the SimpleAI that does not take actions for now
player = agent_jack.Agent(env=env,
                          player_id=player_id,
                          n_actions=n_actions,
                          replay_buffer_size=replay_buffer_size,
                          batch_size=batch_size,
                          hidden_size=hidden_size,
                          gamma=gamma,
                          lr=lr,
                          save_memory=True,
                          frame_stacks=frame_stacks,
                          dagger_files=dagger_files)


"""
x = np.arange(episodes)
y = np.zeros(episodes)
for i in range(0, episodes):
    if i < EXP_EPISODES:
        y[i] = glie_a / (glie_a + i)
    else:
        y[i] = 0.1
plt.plot(x, y)
plt.show()
"""
# Housekeeping
states = []
win1 = 0
frames_list = []
total_frames = 0
rewards_list = []
for i in range(0, episodes):
    done = False
    if i < EXP_EPISODES - 1:
        eps = glie_a / (glie_a + i)
    else:
        eps = 0.1
    obs = env.reset()
    frames = 0
    rewards = 0
    while not done:
        # action1 is zero because in this example no agent is playing as player 0
        action1 = player.get_action(obs, eps)
        ob1, rew1, done, info = env.step(action1)

        player.store_transition(obs, action1, ob1, rew1, done)
        player.update_network()
        obs = ob1
        if args.housekeeping:
            states.append(ob1)
        # Count the wins
        rewards += rew1
        if rew1 == 10:
            win1 += 1
        if not args.headless:
            env.render()
        if done:
            observation = env.reset()
            plt.close()  # Hides game window
            if args.housekeeping:
                plt.plot(states)
                plt.legend(["Player", "Opponent", "Ball X", "Ball Y", "Ball vx", "Ball vy"])
                plt.show()
                states.clear()

            rewards_list.append(rewards)
            if i % 100 == 0 and i > 99:
                rew_avg = round(np.average(rewards_list[i - 99: i]), 2)
                print(f"episode {i} over. Average reward {rew_avg}. Total wins: {win1}. "
                      f"Frames {total_frames} with eps {eps}")

        frames += 1
        total_frames += 1

        if total_frames == 10000:
            print(f"Model saved weights_Jack-v{3}_{total_frames}.mdl")
            torch.save(player.policy_net.state_dict(),
                       f"weights_Jack-v{3}_{total_frames}.mdl")

        if total_frames % 100000 == 0:
            print(f"Model saved weights_Jack-v{3}_{total_frames}.mdl")
            torch.save(player.policy_net.state_dict(),
                       f"weights_Jack-v{3}_{total_frames}.mdl")

    player.update_target_network()
    frames_list.append(frames)
    if i % 1000 == 0 and i > 0:
        x = np.arange(len(rewards_list))
        plt.plot(x, rewards_list)
        plt.xlabel(f"Number of episodes")
        plt.ylabel(f"Reward")
        plt.savefig(f"episodes_{i}.png")
