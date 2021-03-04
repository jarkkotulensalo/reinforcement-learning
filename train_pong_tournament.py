"""
This is an example on how to use the two player Wimblepong environment with one
agent and the SimpleAI
"""
import argparse
import matplotlib.pyplot as plt
import gym
import numpy as np
import warnings
import yaml

from agent import Agent
from utils import calc_glie, get_dagger_files, plot_rewards, plot_exploration_strategy


warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--housekeeping", action="store_true", help="Plot, player and ball positions and velocities at the end of each episode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
parser.add_argument('--config', default='config_tournament.yaml')
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
NUM_ACTIONS = 3

# Number of episodes/games to play
config = yaml.load(open(args.config), Loader=yaml.FullLoader)

# General params
num_episodes = config['num_episodes']  # 100000
update_opponent_win_rate = config['update_opponent_win_rate']
load_path = config['path_pretrained_model']
use_dagger = config['use_dagger']

# Agent params
agent_config = config['agent_params']
batch_size = agent_config['batch_size']
hidden_size = agent_config['fc_hidden_size']
gamma = agent_config['reward_gamma']
replay_memory = agent_config['replay_memory']
num_frame_stacks = agent_config['num_frame_stacks']
target_network_update_frequency = agent_config['target_network_update_frequency']

# Exploration - exploitation epsilon params
epsilon_config = agent_config['epsilon']
EXP_EPISODES = epsilon_config['num_exp_episodes']
exp_end = epsilon_config['end']

# Optim params
optim_config = agent_config['optim_params']
lr = optim_config['lr']
momentum = optim_config['momentum']
eps = optim_config['eps']

glie_a = round(0.1 / 0.9 * EXP_EPISODES, 0)
dagger_files = get_dagger_files(use_dagger)

# dagger_files = None
# Define the player
player_id = 1
opponent_id = 2
# Set up the player here. We used the SimpleAI that does not take actions for now
player = Agent(env=env,
              player_id=player_id,
              optim_params=optim_config,
              n_actions=NUM_ACTIONS,
              replay_buffer_size=replay_memory,
              batch_size=batch_size,
              hidden_size=hidden_size,
              gamma=gamma,
              save_memory=True,
              frame_stacks=num_frame_stacks,
              dagger_files=dagger_files,
              double_dqn=True,
              load_path=load_path
              )

opponent = Agent(env=env,
              player_id=player_id,
              optim_params=optim_config,
              n_actions=NUM_ACTIONS,
              replay_buffer_size=replay_memory,
              batch_size=batch_size,
              hidden_size=hidden_size,
              gamma=gamma,
              save_memory=False,
              frame_stacks=num_frame_stacks,
              dagger_files=dagger_files,
              double_dqn=True,
              load_path=load_path
              )

plot_exploration_strategy(num_episodes, EXP_EPISODES, glie_a, exp_end)

env.set_names(player.get_name(), "opponent")
# Housekeeping
states = []
total_wins1 = 0
frames_list = []
frames_avg_list = []
total_frames = 0
rewards_list = []
rewards_avg_list = []
win_rate_list = []
for episode_num in range(0, num_episodes):
    done = False
    eps = calc_glie(episode_num, EXP_EPISODES, glie_a, exp_end)
    obs1, obs2 = env.reset()
    frames = 0
    rewards = 0
    while not done:
        # action1 is zero because in this example no agent is playing as player 0
        action1 = player.get_action(obs1, eps)
        action2 = opponent.get_action(obs2)
        (next_ob1, next_ob2), (rew1, rew2), done, info = env.step((action1, action2))

        player.store_transition(obs1, action1, next_ob1, rew1, done)
        player.update_network()
        obs1 = next_ob1
        obs2 = next_ob2
        if args.housekeeping:
            states.append(next_ob1)
        # Count the wins
        rewards += rew1
        if rew1 == 10:
            total_wins1 += 1
        if not args.headless:
            env.render()

        if total_frames > 79999:
            if total_frames % target_network_update_frequency == 1:
                # print(f"Updated target network at {total_frames} frames.")
                player.update_target_network()

        frames += 1
        total_frames += 1
        if done:
            # print(f"reward is {rew1}")
            observation = env.reset()
            player.reset()
            opponent.reset()
            plt.close()  # Hides game window
            if args.housekeeping:
                plt.plot(states)
                plt.legend(["Player", "Opponent", "Ball X", "Ball Y", "Ball vx", "Ball vy"])
                plt.show()
                states.clear()

            rewards_list.append(rew1)
            frames_list.append(frames)
            if episode_num % 200 == 0 and episode_num > 0:
                rew_avg = round(np.average(rewards_list[episode_num - 199: episode_num]), 2)
                frames_avg = round(np.average(frames_list[episode_num - 199: episode_num]), 2)
                rewards_avg_list.append(rew_avg)
                frames_avg_list.append(frames_avg)
                win_rate = round(total_wins1/200, 3)
                win_rate_list.append(win_rate)

                print(f"After {episode_num} episode, average reward {rew_avg}, win rate: {win_rate}, avg frames {frames_avg}, eps {round(eps, 3)}")
                total_wins1 = 0

                # When winning gets too easy, upgrade opponent
                if win_rate > update_opponent_win_rate:
                    print(f"Saving model and upgrading opponent.")
                    player.save_model(total_frames=total_frames,
                                      num_frame_stacks=num_frame_stacks)
                    opponent.load_model(total_frames=total_frames,
                                        num_frame_stacks=num_frame_stacks)

        # Save model to track learning
        if total_frames % 200000 == 0 and episode_num > 0:
            player.save_model(total_frames=total_frames,
                              num_frame_stacks=num_frame_stacks)

    if episode_num % 5 == 4:
        env.switch_sides()

    plot_rewards(episode_num, rewards_avg_list, frames_avg_list)

player.save_model(num_frame_stacks, total_frames)
