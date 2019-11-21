import torch
import torch.nn.functional as F

class Policy(torch.nn.Module):
    def init(self, state_space, action_space):
        super().__init__()
        #määrittele policy verkko

        self.fc1 = torch.nn.Linear(state_space, action_space)
        self.init_weights

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        return x


class Agent(object):
    def __init__(self, policy):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = policy.to(self.device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr = 0.01)
        self.batch_size = 1
        self.gamma = 0.8

    def episode_finished(self, episode_number):
        return 0

    def load_model(path):
        state_dict = torch.load(path)
        self.policy.load_state_dict(state_dict)

    def get_action(self, observation, evaluation =False):
        x = featurize(observation)
        return 0
    def reset():
        print("yes")

    def get_name():
        return "jerkko"
