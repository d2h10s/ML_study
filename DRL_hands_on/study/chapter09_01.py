import numpy as np
from tensorboardX import SummaryWriter
from torch import nn
import gym

gamma = .99
lr = .01
epsiodes_to_train = 4

class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)
    
    def calc_qvals(rewards):
        res = []
        sum_r = 0.0
        for r in reversed(rewards):
            sum_r = r+gamma*sum_r
            res.append(sum_r)
        return list(reversed(res))


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    writer = SummaryWriter(comment='-cartpole-reinforce')
    
    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True)
    

