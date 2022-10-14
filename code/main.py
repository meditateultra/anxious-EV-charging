import argparse
import numpy as np
import torch
import pandas as pd
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import random
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
import os


def SampleFromNormalDistribution(mu, sigma, size, min, max):
    times = 0; result = 0
    while times < size:
        sample = np.random.normal(mu, sigma)
        if (sample >= min) and (sample <= max):
            result = sample
            times += 1
    return result


def placeSelection(t):
    mu_arr = [16.8, 9.2, 11.6]
    sigma_arr = [3.6, 2.9, 3.6]
    home_prob = norm.pdf(t, mu_arr[0], sigma_arr[0])
    office_prob = norm.pdf(t, mu_arr[1], sigma_arr[1])
    public_prob = norm.pdf(t, mu_arr[2], sigma_arr[2])
    sum_prob = home_prob + office_prob + public_prob
    home_prob = home_prob / sum_prob
    office_prob = office_prob / sum_prob
    public_prob = public_prob / sum_prob
    place_index = [0, 1, 2]  # 0-home, 1-office, 2-public
    place = int(np.random.choice(place_index, 1, [home_prob, office_prob, public_prob]))
    mu_dep = [9.8, 16.4, 15.6]
    sigma_dep = [3.2, 3.1, 3.7]
    return place, mu_dep[place], sigma_dep[place]


def depatureSim(td_init, k1_init, t_a, times, place, mu, sigma, t_x):
    tx_temp = t_x
    if tx_temp > 24: tx_temp -= 24
    if times == 0:  # change the depature time and soc_d
        flag = 0
        while flag == 0:
            t_d = int(round(np.random.normal(mu, sigma)))
            k1 = SampleFromNormalDistribution(0.9, 0.1, 1, 0.85, 0.95)
            if ((abs(tx_temp - t_a) > 4) & (tx_temp < t_d) & (t_d <= 24) & (t_d > 0)) | ((abs(tx_temp - t_a) <= 4) & (t_d > tx_temp) & (t_d <= 24) & (t_d > 0)) | ((abs(tx_temp - t_a) <= 4) & (t_a > t_d) & (tx_temp > t_d) & (abs(t_d-t_a) > 5) & (t_d <= 24) & (t_d > 0)):
                flag = 1
    else:
        t_d = td_init
        k1 = k1_init
    return t_d, k1
# t_d, k1 = depatureSim(td_init=10, k1_init=0.9, t_a=23, times=0, place=2, mu=9.8, sigma=3.2, t_x=23)
# print(t_d)


# generate soc_x and t_x
def anxiousGenerate(t_d, t_a, k1, t_x):
    t_charge = 1  # init
    # flag = 0
    # while flag == 0:
    #     t_x = int(round(random.uniform(1, 4))) + t_a
    #     flag = 1
    #     if ((t_x <= 24) & (t_a < t_d) & (t_x > t_d)) | ((t_x > 24) & (t_a > t_d) & (t_x-24 > t_d)):
    #         flag = 0
    if t_a > t_d:
        t_charge = (24-t_a) + t_d
    if t_a < t_d:
        t_charge = t_d - t_a
    tx_interval = t_x - t_a
    k2 = SampleFromNormalDistribution(9, 1, 1, 6, 12)
    nominator = k1 * (math.exp(-k2 * tx_interval / t_charge) - 1)
    denominator = math.exp(-k2)-1
    soc_x = nominator / denominator
    if t_x > 24: t_x -= 24
    return t_x, soc_x
# t_x, soc_x = anxiousGenerate(t_d=22, t_a=19, k1=0.9)
# print(t_x)


def socSim(soc_bef, action, mu):
    action_actual = np.ndarray(shape=(1,), buffer=np.array([0.0]))
    # action = 0
    soc = soc_bef + (action * mu)
    if soc > 1:
        surplus = abs(soc - 1)
        gap = abs(1 - soc_bef)
        action_actual = action * gap / (gap + surplus)
        soc = 1
    if soc < 0:
        surplus = abs(soc)
        gap = abs(soc_bef)
        action_actual = action * gap / (gap + surplus)
        soc = 0
    if (soc > 0) & (soc < 1):
        action_actual = action
    return soc, action_actual
# soc, action = socSim(0.9, np.array(0.2), 0.98)
# print(soc)


class Env:
    def __init__(self, ):
        super(Env, self).__init__()

    # simulate user's charging behavior
    def behaviorSim(self, data, t_index, t_a, times, place_info, td_init, k1_init, tx_init, socx):
        t = int(data[t_index][0])  # current time (1-24)
        # set depature time(t_d) and depature soc(soc_d)
        if times == 0:
            place, mu, sigma = placeSelection(t_a)
            place_info = [place, mu, sigma]
            t_d, k1 = depatureSim(td_init=0, k1_init=0, t_a=t_a, times=times, place=place, mu=mu, sigma=sigma, t_x=tx_init)
            t_x, soc_x = anxiousGenerate(t_d, t_a, k1, tx_init)
        else:
            t_d, k1 = depatureSim(td_init=td_init, k1_init=k1_init, t_a=t, times=times, place=place_info[0], mu=place_info[1], sigma=place_info[2], t_x=tx_init)
            t_x = tx_init; soc_x = socx
        return t, t_x, t_d, soc_x, k1, place_info

    def getSoc(self, soc_bef, action, mu):
        soc, action = socSim(soc_bef, action, mu)
        return soc, action

    def state(self, data, start_point, t_index, t_x, t_d, soc, soc_x, soc_d):
        state_lst = []
        for i in range(t_index-start_point, t_index+1):
            state_lst.append(data[i][1])
        info = [t_x, t_d, soc, soc_x, soc_d]
        for i in info: state_lst.append(i)
        return state_lst

    def calculateReward(self, t, t_x, t_d, t_a, action, price, soc_d, soc, kp=7, kx=15, kd=35):
        t_now = t; t_anx = t_x; t_dep = t_d
        r = np.ndarray(shape=(1,), buffer=np.array([0.0]))
        r_anx = np.ndarray(shape=(1,), buffer=np.array([0.0]))
        r_price = np.ndarray(shape=(1,), buffer=np.array([0.0]))

        # if (t_anx <= t_dep) & (t_now < t_x):
        if (t_a-t_anx > 4) & (t_dep < t_a):  # t_x and t_d are both on the next day
            t_anx += 24; t_dep += 24;
            if t_now < t_a: t_now += 24  # t is on the next day
        if (t_anx - t_a <= 4) & (t_dep < t_a):
            t_dep += 24  # t_d is on the next day
            if t_now < t_a: t_now += 24  # t is on the next day

        if t_now < t_anx:
            r_price = -kp * action * price * 0.001 * 150
            r = r_price
        if (t_now >= t_anx) & (t_now < t_dep):
            r_price = -kp * action * price * 0.001 * 150
            r_anx = -kx * max((soc_x - soc), 0) ** 2 # price & TA
            r = r_price + r_anx
        if t_now == t_dep:
            r_anx = -kd * max((soc_d - soc), 0) ** 2  # price & RA
            r = r_anx
        return r, r_anx, r_price


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='hidden size (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=10000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')

parser.add_argument('--clr', type=float, default=0.01, metavar='G',
                    help='critic learning rate (default: 0.0003)')
parser.add_argument('--plr', type=float, default=0.001, metavar='G',
                    help='policy learning rate (default: 0.0003)')
parser.add_argument('--alphalr', type=float, default=0.01, metavar='G',
                    help='alpha learning rate (default: 0.2)')

args = parser.parse_args()

random.seed()
np.random.seed()
# torch.manual_seed(args.seed) # random action

data = pd.read_excel('..\\data\\priceSim.xlsx', engine='openpyxl')
data = data.to_numpy()
env = Env()
state = torch.randn(1, 53)
action = torch.tensor([0.])
next_state = torch.randn(1, 53)

agent = SAC(state.shape[1], action, args)
memory = ReplayMemory(args.replay_size, args.seed)
start_point = 47
total_numsteps = 0
updates = 0
episode_times = 1550
episode_r = []; epoch_price = []; epoch_anx = []
cr1_lst = []; cr2_lst = []; policy_lst = []; alpha_lst = []
episode_reward = 0; anx_reward = 0; price_reward = 0

for i_episode in range(1, episode_times+1):
    ##########################  Start to Charge ##########################################
    episode_steps = 0; t_a = 0;
    done = False
    random.seed()
    t_index = random.randint(start_point, len(data)-25)  # current time index
    times = 0; depart_flag = 0; reward_exempt = 0
    episode_reward = np.array([0.0], dtype='f8')
    anx_reward = np.array([0.0], dtype='f8')
    price_reward = np.array([0.0], dtype='f8')
    while depart_flag == 0:
        if times == 0:
            ########################## S_t ##########################################
            t_a = int(data[t_index][0])
            t_x = int(round(random.uniform(1, 4))) + t_a
            # soc = SampleFromNormalDistribution(0.5, 0.1, 1, 0.2, 0.8)
            soc = np.random.uniform(0, 0.95)
            t, t_x, t_d, soc_x, soc_d, place_info = env.behaviorSim(data, t_index, t_a, times, [], 0, 0, t_x, 0)
        else:
            t = tn; t_d = t_dn; t_x = t_xn; soc_x = soc_xn; soc_d = soc_dn;
        state_lst = env.state(data, start_point, t_index, t_x, t_d, soc, soc_x, soc_d)
        state = np.array(state_lst, dtype='f8')
        action = agent.select_action(state)
        socn, action = env.getSoc(soc, action, mu=0.98)
        soc = socn
        reward, anx, price = env.calculateReward(t, t_x, t_d, t_a, action, state_lst[47], soc_d, soc)
        episode_reward += reward; anx_reward += anx; price_reward += price
        if t == t_d:
            depart_flag = 1
            break
                 ########################## S_t+1 ##########################################
        t_index += 1; times = 1
        if t_index >= len(data)-25:
            depart_flag = 1; reward_exempt = 1
            break
        tn, t_xn, t_dn, soc_xn, soc_dn, place_info = env.behaviorSim(data, t_index, t_a, times, place_info, t_d, soc_d, t_x, soc_x)
        state_next_lst = env.state(data, start_point, t_index, t_xn, t_dn, socn, soc_xn, soc_dn)
        state_next = np.array(state_next_lst, dtype='f8')
        memory.push(state, action, reward, state_next, done)
        total_numsteps += 1

    if reward_exempt == 0:
        # episode_reward = episode_reward / i_episode
        # anx_reward = anx_reward / i_episode
        # price_reward = price_reward / i_episode

        print("episode_reward:", episode_reward, "price reward:", price_reward, "anx_reward:", anx_reward)
        episode_r.append(episode_reward.copy())
        epoch_price.append(price_reward.copy())
        epoch_anx.append(anx_reward.copy())

    if len(memory) > args.batch_size:
        for i in range(args.updates_per_step):  # each training tep
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha \
                = agent.update_parameters(memory, args.batch_size, updates)
            cr1_lst.append(critic_1_loss)
            cr2_lst.append(critic_1_loss)
            policy_lst.append(policy_loss)
            alpha_lst.append(alpha)
            updates += 1
    episode_steps += 1

fig, rplt = plt.subplots(3)
# rplt[0].plot(range(len(episode_r)), np.array(episode_r), 'r')
# rplt[0].set(xlabel='Training episodes', ylabel='Episode reward')
# rplt[1].plot(range(len(episode_r)), np.array(epoch_price))
# rplt[1].set(xlabel='Training episodes', ylabel='Price reward')
# rplt[2].plot(range(len(episode_r)), np.array(epoch_anx))
# rplt[2].set(xlabel='Training episodes', ylabel='Anxiety reward')

fig, rplt0 = plt.subplots()
rplt0.set_ylim(-200,0)
rplt0.plot(range(len(episode_r)), np.array(episode_r), 'r')
rplt0.set(xlabel='Training episodes', ylabel='Episode reward')
fig, rplt1 = plt.subplots()
rplt1.set_ylim(-100,50)
rplt1.plot(range(len(episode_r)), np.array(epoch_price))
rplt1.set(xlabel='Training episodes', ylabel='Price reward')
fig, rplt2 = plt.subplots()
rplt2.set_ylim(-100,0)
rplt2.plot(range(len(episode_r)), np.array(epoch_anx))
rplt2.set(xlabel='Training episodes', ylabel='Anxiety reward')

fig, ax = plt.subplots(3)
ax[0].plot(range(len(cr1_lst)), np.array(cr1_lst))
ax[1].plot(range(len(cr1_lst)), np.array(cr2_lst))
ax[2].plot(range(len(cr1_lst)), np.array(policy_lst))
if not os.path.exists("..\\model\\trainedModel.pb"):
    torch.save(agent.policy.state_dict(), "..\\model\\trainedModel.pb")

        ##########################   Simulation   ##########################################

priceSim = pd.read_excel('..\\data\\reserve_simData.xlsx', engine='openpyxl', header = None)
priceSim = priceSim.to_numpy()
time = [i for i in range(1, 168)]
td_sim = [9, 17, 32, 42, 57, 64, 79, 89, 105, 115, 131, 136, 154, 162, 167]  # depature time
ta_sim = [1, 11, 19, 34, 43, 58, 66, 81, 91, 106, 116, 132, 137, 156, 163]  # start charging time
tx_sim = [2, 13, 21, 36, 44, 59, 68, 82, 92, 108, 119, 133, 140, 157, 164]  # anxious time
socd_sim = []; socx_sim = []
charge_interval = []
for i in range(len(ta_sim)):
    k1 = SampleFromNormalDistribution(0.9, 0.1, 1, 0.85, 0.95)
    k2 = SampleFromNormalDistribution(9, 1, 1, 6, 12)
    socd_sim.append(k1)
    interval = tx_sim[i] - ta_sim[i]
    t_charge = td_sim[i] - ta_sim[i]
    charge_interval.append(t_charge)
    nominator = k1 * (math.exp(-k2 * interval / t_charge) - 1)
    denominator = math.exp(-k2) - 1
    socx_sim.append(nominator / denominator)

soc_sim = [0.5]  # initial soc
index = 47 + ta_sim[0]
soc_now = soc_sim[0]
action_lst = []
iter_times = 0

while iter_times < 15:
    for i in range(charge_interval[iter_times]):
        stateSim = env.state(priceSim, 47, index, (tx_sim[iter_times] % 24), (td_sim[iter_times] % 24), soc_now, socx_sim[iter_times], socd_sim[iter_times])
        stateSim = np.array(stateSim, dtype='f8')
        action = agent.select_action(stateSim)
        action = action.item()
        soc_temp = soc_now
        soc_now += action*0.98
        if soc_now > 1:
            action = action * (1-soc_temp)/(soc_now-soc_temp)
            soc_now = 1
        if soc_now < 0:
            action = action * soc_temp / (soc_temp - soc_now)
            soc_now = 0
        action_lst.append(action)
        soc_sim.append(soc_now)
        index += 1
    for k in range(len(td_sim)):
        if len(soc_sim) == td_sim[k]:
            departFlag = 1
            time_index = k
    if (departFlag == 1) & (time_index != 14):
        for j in range(td_sim[time_index], ta_sim[time_index+1]):
            action = -0.05
            action_lst.append(action)
            soc_now += action
            soc_sim.append(soc_now)
            index += 1
    if time_index == 14:
        break
    iter_times += 1

price = priceSim[48:215, 1]
max_value = np.max(price) + 20
min_value = np.min(price) - 20
price_norm = []
for i in price:
    price_norm.append((i - min_value) / (max_value - min_value))

fig, ax1 = plt.subplots(figsize=(10, 5))
for i in range(len(ta_sim)):
    t_home = []; t_office = []; t_public = [];t_driving = []
    rate = []
    # the x-axis of histogram
    if i < 14:
        for j in range(td_sim[i], ta_sim[i+1]+1): t_driving.append(j)
    if i % 2 == 0:
        for j in range(ta_sim[i], td_sim[i]+1):
            if j != td_sim[len(td_sim)-1]: t_home.append(j)
    if (i % 2 == 1) & (i != 11) & (i != 13):
        for j in range(ta_sim[i], td_sim[i]+1): t_office.append(j)
    if (i == 11) | (i == 13):
        for j in range(ta_sim[i], td_sim[i] + 1): t_public.append(j)

    # the y-axis of histogram
    if len(t_home) != 0:
        for j in t_home: rate.append(action_lst[j - 1])
        if i == 0:
            ax1.bar(np.array(t_home), np.array(rate), color='lightskyblue', label='home')
        else: ax1.bar(np.array(t_home), np.array(rate), color='lightskyblue')
    rate.clear()
    if len(t_driving) != 0:
        for j in t_driving: rate.append(action_lst[j - 1])
        if i == 0:
            ax1.bar(np.array(t_driving), np.array(rate), color='tab:gray', label='driving')
        else: ax1.bar(np.array(t_driving), np.array(rate), color='tab:gray')
    rate.clear()

    if len(t_public) != 0:
        for j in t_public: rate.append(action_lst[j - 1])
        if i == 11:
            ax1.bar(np.array(t_public), np.array(rate), color='darksalmon', label='public')
        else: ax1.bar(np.array(t_public), np.array(rate), color='darksalmon')
    rate.clear()
    if len(t_office) != 0:
        for j in t_office: rate.append(action_lst[j - 1])
        if i == 1:
            ax1.bar(np.array(t_office), np.array(rate), color='goldenrod', label='office')
        else: ax1.bar(np.array(t_office), np.array(rate), color='goldenrod')
    rate.clear()

ax1.legend(loc=0)
ax1.set(xlabel='Time(h)', ylabel='charging power')

fig, sim = plt.subplots(figsize=(10, 5))
sim.plot(range(len(soc_sim)), np.array(soc_sim), 'g', label='SoC')
sim.set(ylabel='SoC')
sim.legend(loc='upper left')
axsim = sim.twinx()
axsim.plot(range(len(soc_sim)), np.array(price_norm), 'r', label='price')
axsim.legend(loc='upper right')
axsim.set(xlabel='time', ylabel='Price')
plt.show()