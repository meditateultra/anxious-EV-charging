import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from model.utils import *

class Env:
    def __init__(self, data_path, args):
        super(Env, self).__init__()
        self.origin_data = pd.read_excel(data_path, engine='openpyxl').to_numpy()
        self.data = MaxMinNormalization(self.origin_data, 1)
        self.args = args

        self.start_point = 47
        self.t_index = random.randint(self.start_point, len(self.data) - 100)

        self.state_list = []
        self.state = np.zeros(shape=53, dtype='f8')
        self.done = False

        self.t_a = int(self.data[self.t_index][0])
        self.t = self.t_a
        self.soc = float(np.random.uniform(0, 0.95))
        place, mu, sigma = self.placeSelection(self.t_a)
        self.place_info = [place, mu, sigma]

        self.t_d, self.soc_d = self.depatureSim()
        self.k2 = 0.01
        self.t_x = self.anxiousTime()
        self.soc_x = self.anxiousGenerate()
        self.state = self.getState()

    def reset(self):
        self.start_point = 47
        self.t_index = random.randint(self.start_point, len(self.data) - 100)
        self.state_list = []
        for i in range(self.t_index - self.start_point, self.t_index + 1):
            self.state_list.append(self.data[i][1])

        self.state = np.zeros(shape=53, dtype='f8')

        self.done = False

        self.t_a = int(self.data[self.t_index][0])
        self.t = self.t_a
        self.soc = float(np.random.uniform(0, 0.95))

        place, mu, sigma = self.placeSelection(self.t_a)
        self.place_info = [place, mu, sigma]

        self.t_d, self.soc_d = self.depatureSim()
        self.k2 = 0.01
        self.t_x = self.anxiousTime()
        self.soc_x = self.anxiousGenerate()
        self.state = self.getState()
        return self.state

    def step(self, action):
        socn, action = self.getSoc(action, mu=0.98)
        self.soc = socn
        reward_tuple = self.calculateReward(action)
        self.t_index += 1
        self.t = int(self.data[self.t_index][0])
        self.soc_x = self.anxiousGenerate()
        self.state = self.getState()
        if self.t == self.t_d:
            self.done = True
        return self.state, reward_tuple, action, self.done

    def placeSelection(self, t):
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

    def depatureSim(self):
        k1 = 0.0
        t_d = 0
        mu = self.place_info[1]
        sigma = self.place_info[2]
        flag = 0
        while flag == 0:
            t_d = int(round(np.random.normal(mu, sigma)))
            k1 = SampleFromNormalDistribution(0.9, 0.1, 1, 0.85, 0.95)
            if t_d > 24 or t_d < 1:
                continue
            if t_d < self.t_a and t_d + 24 - self.t_a >= 2:
                flag = 1
            elif t_d - self.t_a >= 2:
                flag = 1
        if t_d > 24:
            t_d -= 24
        return t_d, k1

    def anxiousTime(self):
        if self.t_d > self.t_a:
            if self.t_d - self.t_a <= 4:
                t_x = self.t_d - int(round(random.uniform(1, self.t_d - self.t_a - 1)))
            else:
                t_x = self.t_d - int(round(random.uniform(1, 4)))
        else:
            if self.t_d - self.t_a + 24 <= 4:
                t_x = self.t_d + 24 - int(round(random.uniform(1, self.t_d - self.t_a + 24 - 1)))
            else:
                t_x = self.t_d + 24 - int(round(random.uniform(1, 4)))
        if t_x > 24:
            t_x -= 24
        return t_x

    def anxiousGenerate(self):
        t_now = 24 if self.t % 24 == 0 else self.t % 24
        t_anx = 24 if self.t_x % 24 == 0 else self.t_x % 24
        t_dep = 24 if self.t_d % 24 == 0 else self.t_d % 24
        if t_now < self.t_a:
            t_now += 24
        if t_dep < self.t_a:
            t_dep += 24
        if t_anx < self.t_a:
            t_anx += 24

        if t_now < t_anx:
            return 0.0
        t_charge = t_dep - self.t_a
        tx_interval = t_now - self.t_a
        nominator = self.soc_d * (math.exp(-self.k2 * tx_interval / t_charge) - 1)
        denominator = math.exp(-self.k2) - 1
        soc_x = nominator / denominator
        return soc_x

    def getSoc(self, action, mu):
        action_actual = np.ndarray(shape=(1,), buffer=np.array([0.0]))
        soc = self.soc + (action * mu)
        if soc > 1:
            surplus = abs(soc - 1)
            gap = abs(1 - self.soc)
            action_actual = action * gap / (gap + surplus)
            soc = 1
        if soc < 0:
            surplus = abs(soc)
            gap = abs(self.soc)
            action_actual = action * gap / (gap + surplus)
            soc = 0
        if (soc > 0) & (soc < 1):
            action_actual = action
        return float(soc), action_actual

    def getState(self):
        self.state_list = []
        for i in range(self.t_index - self.start_point, self.t_index + 1):
            # state_lst.append(0.1)
            self.state_list.append(self.data[i][1])
        info = [self.t_x / 24.0, self.t_d / 24.0, self.soc, self.soc_x, self.soc_d]
        for i in info:
            self.state_list.append(i)
        return np.array(self.state_list, dtype='f8')

    def calculateReward(self, action, kp=1.5, kx=17, kd=35):
        price = self.origin_data[self.t_index][1]
        t_now = 24 if self.t % 24 == 0 else self.t % 24
        t_anx = 24 if self.t_x % 24 == 0 else self.t_x % 24
        t_dep = 24 if self.t_d % 24 == 0 else self.t_d % 24
        r = np.ndarray(shape=(1,), buffer=np.array([0.0]))
        r_anx = np.ndarray(shape=(1,), buffer=np.array([0.0]))
        r_price = np.ndarray(shape=(1,), buffer=np.array([0.0]))

        if t_now < self.t_a:
            t_now += 24
        if t_dep < self.t_a:
            t_dep += 24
        if t_anx < self.t_a:
            t_anx += 24

        if t_now < t_anx:
            r_price = -kp * action * price
            r = r_price
        elif t_now == t_dep - 1:
            r_price = -kp * action * price
            r_anx = -kd * max((self.soc_d - self.soc), 0) ** 2
            r = r_price + r_anx
        elif t_anx <= t_now < t_dep:
            r_price = -kp * action * price
            r_anx = -kx * max((self.soc_x - self.soc), 0) ** 2  # price & TA
            r = r_price + r_anx
        return r, r_anx, r_price

    def simulation(self, agent):
        self.reset()
        priceSim = pd.read_excel('price/testPrice.xlsx', engine='openpyxl', header=None)
        priceSim = priceSim.to_numpy()
        price = priceSim[48:215, 1]
        priceSim = MaxMinNormalization(priceSim, 1)
        self.data = priceSim
        time = [i for i in range(1, 168)]
        td_sim = [9, 17, 32, 42, 57, 64, 79, 89, 105, 115, 131, 136, 154, 162, 167]  # depature time
        ta_sim = [1, 11, 19, 34, 43, 58, 66, 81, 91, 106, 116, 132, 137, 156, 163]  # start charging time
        tx_sim = [7, 14, 30, 38, 54, 62, 76, 88, 101, 112, 128, 133, 150, 160, 165]  # anxious time
        socd_sim = []
        k2_sim = []
        charge_interval = []

        for i in range(len(ta_sim)):
            k1 = SampleFromNormalDistribution(0.9, 0.1, 1, 0.85, 0.95)
            k2 = SampleFromNormalDistribution(9, 1, 1, 6, 12)
            socd_sim.append(k1)
            k2_sim.append(k2)
            interval = tx_sim[i] - ta_sim[i]
            t_charge = td_sim[i] - ta_sim[i]
            charge_interval.append(t_charge)

        soc_sim = [0.8]  # initial soc
        index = 47 + ta_sim[0]
        action_lst = []
        iter_times = 0
        self.soc = soc_sim[0]

        while iter_times < 15:
            self.t_index = index
            self.t_a = 24 if ta_sim[iter_times] % 24 == 0 else ta_sim[iter_times] % 24
            self.t_d = 24 if td_sim[iter_times] % 24 == 0 else td_sim[iter_times] % 24
            self.t_x = 24 if tx_sim[iter_times] % 24 == 0 else tx_sim[iter_times] % 24
            self.t = self.t_a
            self.soc_d = socd_sim[iter_times]
            self.k2 = k2_sim[iter_times]
            self.soc_x = self.anxiousGenerate()
            self.state = self.getState()
            for i in range(charge_interval[iter_times]):
                action = agent.select_action(self.state)
                next_state, _, action, _ = self.step(action)
                action = action.item()
                action_lst.append(action)
                soc_sim.append(self.soc)
                self.state = next_state
                index += 1
            for k in range(len(td_sim)):
                if len(soc_sim) == td_sim[k]:
                    departFlag = 1
                    time_index = k
            if (departFlag == 1) & (time_index != 14):
                for j in range(td_sim[time_index], ta_sim[time_index + 1]):
                    action = -0.05
                    action_lst.append(action)
                    self.soc += action
                    soc_sim.append(self.soc)
                    index += 1
            if time_index == 14:
                break
            iter_times += 1

        max_value = np.max(price)
        min_value = np.min(price)
        price_norm = []

        home_time = []
        home_charge = []
        office_time = []
        office_charge = []
        driving_time = []
        driving_charge = []

        for i in price:
            price_norm.append((i - min_value) / (max_value - min_value))
            # price_norm.append(i)

        fig, ax1 = plt.subplots(figsize=(10, 5))
        for i in range(len(ta_sim)):
            t_home = []
            t_office = []
            t_public = []
            t_driving = []
            rate = []
            if i < 14:
                for j in range(td_sim[i], ta_sim[i + 1]):
                    t_driving.append(j)
                    driving_time.append(j)
            if i % 2 == 0:
                for j in range(ta_sim[i], td_sim[i]):
                    if j != td_sim[len(td_sim) - 1]:
                        t_home.append(j)
                        home_time.append(j)
            if (i % 2 == 1) & (i != 11) & (i != 13):
                for j in range(ta_sim[i], td_sim[i]):
                    t_office.append(j)
                    office_time.append(j)
            if (i == 11) | (i == 13):
                for j in range(ta_sim[i], td_sim[i] + 1):
                    t_public.append(j)

            # the y-axis of histogram
            if len(t_home) != 0:
                for j in t_home:
                    rate.append(action_lst[j - 1])
                    home_charge.append(action_lst[j - 1])
                if i == 0:
                    ax1.bar(np.array(t_home), np.array(rate), color='lightskyblue', label='Home')
                else:
                    ax1.bar(np.array(t_home), np.array(rate), color='lightskyblue')
            rate.clear()
            if len(t_driving) != 0:
                for j in t_driving:
                    rate.append(action_lst[j - 1])
                    driving_charge.append(action_lst[j - 1])
                if i == 0:
                    ax1.bar(np.array(t_driving), np.array(rate), color='tab:gray', label='Driving')
                else:
                    ax1.bar(np.array(t_driving), np.array(rate), color='tab:gray')
            rate.clear()

            if len(t_public) != 0:
                for j in t_public:
                    rate.append(action_lst[j - 1])
                if i == 11:
                    ax1.bar(np.array(t_public), np.array(rate), color='darksalmon', label='public')
                else:
                    ax1.bar(np.array(t_public), np.array(rate), color='darksalmon')
            rate.clear()
            if len(t_office) != 0:
                for j in t_office:
                    rate.append(action_lst[j - 1])
                    office_charge.append(action_lst[j - 1])
                if i == 1:
                    ax1.bar(np.array(t_office), np.array(rate), color='goldenrod', label='Office')
                else:
                    ax1.bar(np.array(t_office), np.array(rate), color='goldenrod')
            rate.clear()

        ax1.legend(loc=0)
        ax1.set(xlabel='Time(hour)', ylabel='Charging Power(%)')
        ax1.set_ylim(-0.2, 0.2)
        ax2 = ax1.twinx()
        ax2.plot(range(120), price_norm[:120], color='red', label='Price')
        ax2.legend(loc='upper right')
        ax2.set(ylabel='Price')
        ax2.set_ylim(0, 1)
        fig.savefig(os.path.join(self.args.save_path, 'pic1.png'))

        fig, sim = plt.subplots(figsize=(10, 5))
        sim.plot(range(120), soc_sim[:120], 'b', label='SoC')
        sim.set(ylabel='SoC')
        sim.legend(loc='upper right')
        sim.set_ylim(0, 1)
        fig.savefig(os.path.join(self.args.save_path, 'pic2.png'))
        plt.show()

        write_list1 = []
        write_list2 = []
        for i in range(len(home_time) + len(office_time) + len(driving_time)):
            if i < len(home_time):
                write_list1.append([home_time[i] - 1, home_charge[i], 1])
            elif len(home_time) <= i < len(home_time) + len(office_time):
                write_list1.append([office_time[i - len(home_time)] - 1,
                                    office_charge[i - len(home_time)], 2])
            else:
                write_list1.append([driving_time[i - len(home_time) - len(office_time)] - 1,
                                    driving_charge[i - len(home_time) - len(office_time)], 3])
        for i in range(len(price_norm)):
            write_list2.append([i, soc_sim[i], price_norm[i]])
        s = pd.DataFrame(np.array(write_list1))
        s.to_csv(os.path.join(self.args.save_path, 'rate.csv'), index=False)
        s = pd.DataFrame(np.array(write_list2))
        s.to_csv(os.path.join(self.args.save_path, 'soc.csv'), index=False)