import numpy as np
import pandas as pd
from gurobipy import *
from numpy import random
import math
import csv


def SampleFromNormalDistribution(mu, sigma, size, min, max):
    times = 0
    result = []
    while times < size:
        sample = random.normal(mu,sigma)
        if (sample >= min) and (sample <= max):
            result.append(sample)
            times += 1
    return result


# soc data generated
dataset = pd.read_csv('D:\A_zxzhang\EV Charging\pytorch-soft-actor-critic\data\gurobiSim_data.csv')
dataset = dataset.to_numpy()
charging_duration = len(dataset) - 47
price_data = np.empty([charging_duration, 54])

size = len(dataset) - 47  # size for one file
soc_t = np.empty([size, 24])
rate = np.empty([size, 24])

for i in range(0, size):
    k = 0
    for j in range(i, 48+i):
        price_data[i][k] = dataset[j][1]
        k = k+1

t_d = []; t_x = []; t_a = []
for i in range(47, len(dataset)):
    t_d.append(int(round(dataset[i][3])))  # depature time from Jan 3rd
    t_a.append(int(dataset[i][0]))  # start time

for i in t_a:
    t_x.append(int(round(random.uniform(0, 4))+i))  # anxious time

k1 = SampleFromNormalDistribution(mu=0.9, sigma=0.1, size=size, min=0.85, max=0.95)  # soc_d
soc_d = k1
k2 = SampleFromNormalDistribution(mu=9, sigma=1, size=size, min=6, max=12)
# soc_a = SampleFromNormalDistribution(mu=0.5, sigma=0.1, size=size, min=0.2, max=0.8)
soc_a = np.random.uniform(0, 0.95, size=size)
soc_x = []; t_charge = []
for i in range(0, size):
    nominator = k1[i] * (math.exp(-k2[i]*(t_x[i]-t_a[i])/(t_d[i]-t_a[i]))-1)
    denominator = math.exp(-k2[i])-1
    soc_temp = nominator/denominator
    if soc_temp == 0:
        soc_temp = 0.9
    soc_x.append(soc_temp)
    # soc_x.append(k1[i]*(math.exp(-k2[i]*(t_x[i]-t_a[i])/(t_d[i]-t_a[i]))-1)/(math.exp(-k2[i])-1))
    t_charge.append(int(t_d[i]-t_a[i]))

price_mean = 16.72  # 6 months' average price at point 1
price4sim = np.empty([size, 24])  # price data for simulation
for i in range(size):
    for j in range(t_charge[i]):
        price4sim[i][j] = dataset[i+j][1]/price_mean


for i in range(size):
    try:
        model = Model("mip1")
        soc_lst = []; gap_lst = []; rate_lst = []
        for j in range(int(t_a[i]), int(t_d[i])):
            # Create variables
            charging_rate = model.addVar(lb=-0.2, ub=0.2, vtype=GRB.CONTINUOUS, name="charging_rate")
            if len(soc_lst) == 0:
                soc_now = soc_a[j] + charging_rate
            else:
                soc_now = soc_lst[j-1-t_a[i]] + charging_rate
            payment = charging_rate * price4sim[i][j-int(t_a[i])]
            soc_gap = soc_d[i] - soc_now
            # Set objective
            if i < t_x[i]:
                model.setObjective(payment, GRB.MINIMIZE)
            else:
                model.setObjective(0.5 * soc_gap + 0.5 * payment, GRB.MINIMIZE)
            # Add constraint:
            model.addConstr(soc_now <= 1, "ueq1")
            model.addConstr(soc_now >= 0, "ueq2")
            # model._vars = model.getVars()
            # model.optimize(mycallback)
            model.optimize()
            if GRB.Callback.SIMPLEX:
                for v in model.getVars():
                    if len(model.getVars()) == t_charge[i]:
                        rate_lst.append(v.x)
            gap_lst.append(soc_gap.getValue())
            soc_lst.append(soc_now.getValue())  # soc_t
            # for v in m.getVars():
            #     if GRB.Callback.SIMPLEX:
            #         rate_lst.append(v.x)  # fetch the rate data
        soc_t[i][0:len(soc_lst)] = np.array(soc_lst)
        rate[i][0:len(rate_lst)] = np.array(rate_lst)

    except GurobiError:
        print('Error reported')

state_data = np.empty([20000, 54])
index = 0; flag = 0
for i in range(size):
    times = 0
    if flag == 1:
        break
    for j in range(t_charge[i]):
        if i+times >= size:
            flag = 1
            break
        state_data[i+index][:] = price_data[i+times][:]  # 0~47 price data
        state_data[i+index][48] = t_x[i]  # 48 anxious time
        state_data[i+index][49] = t_d[i]  # 49 departure time
        state_data[i + index][50] = soc_t[i][j]  # 50 soc_t
        state_data[i + index][51] = soc_x[i]  # 51 soc_x
        state_data[i + index][52] = soc_d[i]  # 52 soc_d
        state_data[i + index][53] = rate[i][j]  # label: charging rate
        index = index + 1
        times = times + 1

state_data = state_data.tolist()  # remove all the zero rows
state_data_final = []
for i in range(len(state_data)):
    list_0 = [0 for i in range(len(state_data[i][:-1]))]
    if state_data[i][:-1] != list_0:
        state_data_final.append(state_data[i])

state_data_array = np.empty([16963, 54])
j = 0
for i in state_data_final:
    state_data_array[j][:] = np.array(i)
    j = j+1

with open('D:\A_zxzhang\EV Charging\pytorch-soft-actor-critic\data\SLData.csv', 'w', newline='') as file:
    mywriter = csv.writer(file, delimiter=',')
    mywriter.writerows(state_data_array)




