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
replay_buffer_size = 200000
batch_size = 32
hidden_size = 512
gamma = 0.99
lr = 2.5e-4
frame_stacks = 2
EXP_EPISODES = 50000
glie_a = round(0.1 / 0.9 * EXP_EPISODES, 0)

# https://towardsdatascience.com/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756
TARGET_UPDATE_FRAMES = 10000
dagger_files = ['./dagger/mem9-1.pickle',
                './dagger/mem7-3.pickle',
                './dagger/mem6-4.pickle',
                './dagger/mem6-5.pickle',
                './dagger/mem25-6.pickle']

# load_path = "./pretrained_models/weights_Jack-v4_3000000.mdl"
load_path = ""

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
                          dagger_files=dagger_files,
                          double_dqn=True,
                          load_path=load_path)

x = np.arange(episodes)
y = np.zeros(episodes)
for i in range(0, episodes):
    if i < EXP_EPISODES:
        y[i] = glie_a / (glie_a + i)
    else:
        y[i] = 0.1

plt.ylabel('Exploration')
plt.xlabel('Number of episodes')
plt.plot(x, y)
plt.savefig(f"exploration_{episodes}.png")

# Housekeeping
states = []
win1 = 0
frames_list = []
frames_avg_list = []
total_frames = 0
rewards_list = []
rewards_avg_list = []
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

        frames += 1
        total_frames += 1
        if done:
            observation = env.reset()
            plt.close()  # Hides game window
            if args.housekeeping:
                plt.plot(states)
                plt.legend(["Player", "Opponent", "Ball X", "Ball Y", "Ball vx", "Ball vy"])
                plt.show()
                states.clear()

            rewards_list.append(rew1)
            frames_list.append(frames)
            if i % 200 == 0 and i > 200:
                rew_avg = round(np.average(rewards_list[i - 199: i]), 2)
                frames_avg = round(np.average(frames_list[i - 199: i]), 2)
                print(f"episode {i} over. Average reward {rew_avg}. Total wins: {win1}. "
                      f"Frames {frames_avg} with eps {round(eps, 3)}")
                rewards_avg_list.append(rew_avg)
                frames_avg_list.append(frames_avg)

        if total_frames % TARGET_UPDATE_FRAMES == 1:
            print(f"Updated target network at {total_frames} frames.")
            player.update_target_network()

        if total_frames == 10000:
            print(f"Model saved weights_Jack-v{1}_{total_frames}.mdl")
            torch.save(player.policy_net.state_dict(),
                       f"pretrained_models\weights_Jack-v{1}_{total_frames}.mdl")

        if total_frames % 100000 == 0:
            print(f"Model saved weights_Jack-v{1}_{total_frames}.mdl")
            torch.save(player.policy_net.state_dict(),
                       f"pretrained_models\weights_Jack-v{1}_{total_frames}.mdl")


    if (i % 10000 == 0 and i > 0) or (i == 1000):
        fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, figsize=(6.4, 4.8 * 2))
        x = np.arange(len(rewards_avg_list))
        x = x * 200
        ax1.plot(x, rewards_avg_list)
        ax1.set_xlabel(f"Number of episodes")
        ax1.set_ylabel(f"Avg. reward for 100 episodes")

        ax2.plot(x, frames_avg_list)
        ax2.set_xlabel(f"Number of episodes")
        ax2.set_ylabel(f"Avg. frame duration for 100 episodes")

        fig.savefig(f"plots\rewards_{i}.png")
        print(f"Learning plot saved after episode {i}.")
