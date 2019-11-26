from utils import ReplayMemory, Transition
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from wimblepong import Wimblepong
import random
from utils import rgb2gray

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class DQN(nn.Module):
    def __init__(self,in_channels,n_channels,n_actions):
        super(DQN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = n_channels, kernel_size = 16, stride = 2,padding = 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels = n_channels, out_channels = 64, kernel_size = 12)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 8)

        self.fc1 = nn.Linear(32*11*11,128)
        self.fc2 = nn.Linear(128,32)
        self.fc3 = nn.Linear(32,8)
        self.fc4 = nn.Linear(8,3)



    def forward(self,x):
        #print("---")
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = self.conv3(x)
        #print(x.shape)
        #print("---")


        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class DQN_agent(object):
    def __init__(self, env, player_id=1, capacity = 100000, batch_size = 128, gamma = 0.98):
        if type(env) is not Wimblepong:
            raise TypeError("I'm not a very smart AI. All I can play is Wimblepong.")
        self.env = env
        # Set the player id that determines on which side the ai is going to play
        self.player_id = player_id
        # Ball prediction error, introduce noise such that SimpleAI reflects not
        # only in straight lines

        self.name = "DQN_agent"

        self.policy_net = DQN(in_channels = 1, n_channels = 32, n_actions=3).to(device)

        self.target_net = DQN(in_channels = 1, n_channels = 32, n_actions=3).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.memory = ReplayMemory(capacity)
        print("using device: {}".format(device))
        self.memory.load_memory('./mem9-1.pickle','./mem7-3.pickle')
        self.batch_size = batch_size
        self.gamma = gamma

    def update_network(self, updates = 1):
        for _ in range(updates):
            self._do_network_update()

    def _do_network_update(self):
        if len(self.memory) < self.batch_size:
            return
        else:
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            non_final_mask = 1-torch.tensor(batch.done, dtype=torch.uint8)
            non_final_next_states = [s for nonfinal,s in zip(non_final_mask,
                                         batch.next_ob1) if nonfinal > 0]
            non_final_next_states = torch.stack(non_final_next_states).to(device)
            state_batch = torch.stack(batch.ob1).unsqueeze(1)

            action_batch = torch.cat(batch.action1).to(device)
            reward_batch = torch.cat(batch.rew1).to(device)
            state_action_values = self.policy_net(state_batch.to(device)).gather(1, action_batch)

            next_state_values = torch.zeros(self.batch_size).to(device)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states.unsqueeze(1).to(device)).max(1)[0].detach()

            expected_state_action_values = reward_batch + self.gamma*next_state_values

            #loss = F.mse_loss()
            loss = F.smooth_l1_loss(state_action_values.squeeze(),
                                expected_state_action_values)

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1e-1, 1e-1)
            self.optimizer.step()

    def update_target_network(self):

        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, ob, action, next_ob, reward, done):
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32)
        next_ob = torch.from_numpy(next_ob).float()
        ob = torch.from_numpy(ob).float()
        self.memory.push(ob, action, next_ob, reward, done)

    def get_name(self):
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def get_action(self, ob=None, eps = 0.05):
        """
        Interface function that returns the action that the agent took based
        on the observation ob
        """
        if random.random() > eps:
            with torch.no_grad():
                ob = torch.from_numpy(ob).float().unsqueeze(0).unsqueeze(0).to(device)
                q_values = self.policy_net(ob)
                return torch.argmax(q_values).item()
        else:
            action = random.randrange(self.env.action_space.n) # TODO test

        return action

    def reset(self):
        # Nothing to done for now...
        return

    def load_policy(self,loadpath):
        self.policy_net.load_state_dict(torch.load(loadpath))
