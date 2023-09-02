import argparse
import os

from model.sac import SAC
from torch.utils.tensorboard import SummaryWriter
from model.replay_memory import ReplayMemory
import random
import matplotlib.pyplot as plt

from model.utils import *
from model.env import Env

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='hidden size (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=5000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=10000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true", help='run on CUDA (default: False)')

parser.add_argument('--clr', type=float, default=0.0003, metavar='G',
                    help='critic learning rate (default: 0.0003)')
parser.add_argument('--plr', type=float, default=0.0001, metavar='G',
                    help='policy learning rate (default: 0.0003)')
parser.add_argument('--alphalr', type=float, default=0.0002, metavar='G',
                    help='alpha learning rate (default: 0.2)')

parser.add_argument('--simulate', action="store_true", help='simulate the EV charging (default: False)')
parser.add_argument('--save_path', type=str, default='run/one/', metavar='N', help='where the result and model save')
parser.add_argument('--policy_path', type=str, default='example/policy.pb', metavar='N',
                    help='the parameters of policy network')

args = parser.parse_args()

random.seed()
np.random.seed()

env = Env('price/trainPrice.xlsx', args)
state = torch.randn(1, 53)
action = torch.tensor([0.])
next_state = torch.randn(1, 53)

agent = SAC(state.shape[1], action, args)

if not args.simulate:
    writer = SummaryWriter(args.save_path)
    memory = ReplayMemory(args.replay_size, args.seed)
    total_numsteps = 0
    updates = 0
    episode_times = 2000
    episode_r = []
    epoch_price = []
    epoch_anx = []
    cr1_lst = []
    cr2_lst = []
    policy_lst = []
    alpha_lst = []
    episode_reward = np.array([0.0], dtype='f8')
    anx_reward = np.array([0.0], dtype='f8')
    price_reward = np.array([0.0], dtype='f8')


    steps = 0

    for i_episode in range(1, episode_times + 1):
        ##########################  Start to Charge ##########################################
        steps = 0
        episode_steps = 0
        done = False
        random.seed()
        state = env.reset()
        episode_reward = np.array([0.0], dtype='f8')
        anx_reward = np.array([0.0], dtype='f8')
        price_reward = np.array([0.0], dtype='f8')
        while not done:
            if args.start_steps > total_numsteps:
                action = np.array([np.random.uniform(-0.2, 0.2)], dtype='f8')
            else:
                action = agent.select_action(state)

            if len(memory) > args.batch_size:
                for i in range(args.updates_per_step):  # each training tep
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha \
                        = agent.update_parameters(memory, args.batch_size, updates)
                    cr1_lst.append(critic_1_loss)
                    cr2_lst.append(critic_1_loss)
                    policy_lst.append(policy_loss)
                    alpha_lst.append(alpha)
                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward_tuple, action, done = env.step(action)
            reward = reward_tuple[0]
            anx = reward_tuple[1]
            price = reward_tuple[2]
            episode_reward += reward_tuple[0]
            anx_reward += reward_tuple[1]
            price_reward += reward_tuple[2]

            memory.push(state, action / 0.2, reward, next_state, float(not done))

            state = next_state
            total_numsteps += 1
            steps += 1

        print("Episode:", i_episode, "episode_reward:", episode_reward, "price reward:", price_reward, "anx_reward:",
              anx_reward)
        writer.add_scalar('reward/sum_reward', episode_reward, i_episode)
        writer.add_scalar('reward/price_reward', price_reward, i_episode)
        writer.add_scalar('reward/anx_reward', anx_reward, i_episode)

        episode_r.append(episode_reward)
        epoch_price.append(price_reward)
        epoch_anx.append(anx_reward)

        episode_steps += 1

    torch.save(agent.policy.state_dict(), os.path.join(args.save_path, 'policy.pb'))
    torch.save(agent.critic.state_dict(), os.path.join(args.save_path, 'critic.pb'))


    agent.policy.load_state_dict(torch.load(os.path.join(args.save_path, 'policy.pb')))
    agent.critic.load_state_dict(torch.load(os.path.join(args.save_path, 'critic.pb')))

    fig, rplt0 = plt.subplots()
    rplt0.plot(range(0, episode_times), np.array(episode_r), 'r')
    rplt0.set(xlabel='Training episodes', ylabel='Episode reward')
    fig.savefig(os.path.join(args.save_path, 'pic4.png'))
    fig, rplt1 = plt.subplots()
    rplt1.plot(range(0, episode_times), np.array(epoch_price))
    rplt1.set(xlabel='Training episodes', ylabel='Price reward')
    fig.savefig(os.path.join(args.save_path, 'pic5.png'))
    fig, rplt2 = plt.subplots()
    rplt2.plot(range(0, episode_times), np.array(epoch_anx))
    rplt2.set(xlabel='Training episodes', ylabel='Anxiety reward')
    fig.savefig(os.path.join(args.save_path, 'pic6.png'))

    # V2G/G2V simulation in a week
    env.simulation(agent)
else:
    os.makedirs(args.save_path, exist_ok=True)
    agent.policy.load_state_dict(torch.load(args.policy_path))
    env.simulation(agent)