import torch
import torch.nn as nn
import torch.nn.functional as F

class DDQN(nn.Module):
    def __init__(self, state_dim, action_dim,hidden_dim=256):
        super(DDQN, self).__init__()


        self.fc1 = nn.Linear(state_dim, hidden_dim)

        self.fc_value = nn.Linear(hidden_dim, 256)
        self.value = nn.Linear(256, 1)


        self.fc_advantages = nn.Linear(hidden_dim, 256)
        self.advantages = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        v = F.relu(self.fc_value(x))
        V = self.value(v)

        a = F.relu(self.fc_advantages(x))
        A = self.advantages(a)

        Q = V + A - torch.mean(A,dim=1, keepdim=True)
        return Q
if __name__ == '__main__':
    state_dim = 12
    action_dim = 2
    net = DDQN(state_dim,action_dim)
    state = torch.randn(1,state_dim)
    output = net(state)
    state = torch.randn(5, state_dim)