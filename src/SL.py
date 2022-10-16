import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.distributions import Normal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
# model parameter
num_inputs = 53
hidden_dim = 128
num_actions = 1


def weights_init_(m):
    if isinstance(m, nn.Linear):
        # m contains various linear function, thus gives gain=1
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        # let all values in tenser(m.bias) be 0
        torch.nn.init.constant_(m.bias, 0)


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.action_scale = torch.tensor(0.2)
        self.action_bias = torch.tensor(0.)
        # construct the connection between hidden_dim(Penultimate layer) to the output layer
        # output contains mean and standard in log form
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        # Clamps all elements in input into the range[min, max]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):  # reparameterization trick
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        # log_prob = normal.log_prob(x_t)
        # # Enforcing Action Bound
        # log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        # log_prob = log_prob.sum(1, keepdim=True)
        # mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action


def load_array(data_arrays, batch_size, is_train=True):  # load and shuffle data
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


state_data = pd.read_csv('D:\A_zxzhang\EV Charging\pytorch-soft-actor-critic\data\stateData.csv')
state_data = state_data.to_numpy()
size = 11776
state_data = state_data[0:size][:]
label_data = pd.read_excel('D:\A_zxzhang\EV Charging\pytorch-soft-actor-critic\data/actionData.xlsx', engine='openpyxl')
label_data = label_data.to_numpy()
label_data = label_data[0:size][:]

net = GaussianPolicy(num_inputs, num_actions, hidden_dim)
state_tensor = torch.randn(size, 53)
label_tensor = torch.randn(size, 1)
for i in range(size):
    state_tensor[i] = torch.tensor(state_data[i])
    label_tensor[i] = torch.tensor(label_data[i])

batch_size = 128
data_iter = load_array((state_tensor, label_tensor), batch_size)
num_epochs = 5000
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.0001)

loss_y = []
for epoch in range(num_epochs):
    running_loss = 0
    for X, y in data_iter:  # need to confirm
        l = loss(net.sample(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        running_loss += l.item()
    print(f'epoch {epoch + 1}, loss {running_loss:f}')
    loss_y.append(running_loss)

torch.save(net.state_dict(), "D:\A_zxzhang\EV Charging\pytorch-soft-actor-critic\model\pretrain.pb") # trained model
plt.figure()
plt.style.use('seaborn-darkgrid')
plt.plot(range(1, 5001), np.array(loss_y), 'r')
plt.xlabel("Training episodes")
plt.ylabel("Train loss")
plt.show()
