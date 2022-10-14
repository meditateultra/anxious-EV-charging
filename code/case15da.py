'''15-bus radial distribution system from Das, Kothari, and Kalam'''
from casadi import *


class DistributionSystem:
    mpc_bus = [
        [1, 3, 0, 0, 0, 0, 1, 1, 0, 11, 1, 1, 1],
        [2, 1, 44.1, 44.991, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [3, 1, 70, 71.4143, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [4, 1, 140, 142.8286, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [5, 1, 44.1, 44.991, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [6, 1, 140, 142.8286, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [7, 1, 140, 142.8286, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [8, 1, 70, 71.4143, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [9, 1, 70, 71.4143, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [10, 1, 44.1, 44.991, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [11, 1, 140, 142.8286, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [12, 1, 70, 71.4143, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [13, 1, 44.1, 44.991, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [14, 1, 70, 71.4143, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [15, 1, 140, 142.8286, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
    ]
    mpc_branch = [
        [1, 2, 1.35309, 1.32349, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [2, 3, 1.17024, 1.14464, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [3, 4, 0.84111, 0.82271, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [4, 5, 1.52348, 1.0276, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [2, 9, 2.01317, 1.3579, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [9, 10, 1.68671, 1.1377, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [2, 6, 2.55727, 1.7249, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [6, 7, 1.0882, 0.734, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [6, 8, 1.25143, 0.8441, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [3, 11, 1.79553, 1.2111, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [11, 12, 2.44845, 1.6515, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [12, 13, 2.01317, 1.3579, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [4, 14, 2.23081, 1.5047, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [4, 15, 1.19702, 0.8074, 0, 0, 0, 0, 0, 0, 1, -360, 360],
    ]

    def __init__(self, root):
        self.opti = Opti()
        self.root = root
        self.P = [self.opti.variable() for i in range(len(self.mpc_branch))]
        self.Q = [self.opti.variable() for i in range(len(self.mpc_branch))]
        self.v = [self.opti.variable() for i in range(len(self.mpc_bus))]
        self.p0 = -self.mpc_bus[root-1][2]
        self.q0 = -self.mpc_bus[root-1][3]

    '''得到需要优化的目标函数'''
    def objective(self):
        for h in range(len(self.mpc_branch)):
            branch = self.mpc_branch[h]
            if branch[0] == self.root:
                self.p0 += self.P[h]
                self.q0 += self.Q[h]
        self.opti.minimize(self.p0)

    '''对于每个bus用基尔霍夫定律，定义约束'''
    def constraint(self):
        for i in range(len(self.mpc_bus)):
            bus = self.mpc_bus[i]
            active_equation = -bus[2]
            reactive_equation = -bus[3]
            if bus[0] != self.root:
                for j in range(len(self.mpc_branch)):
                    branch = self.mpc_branch[j]
                    if branch[0] == bus[0]:
                        active_equation -= self.P[j]
                        reactive_equation -= self.Q[j]
                    if branch[1] == bus[0]:
                        # 算电流
                        l = (self.P[j] ** 2 + self.Q[j] ** 2) / self.v[i] ** 2
                        active_equation += self.P[j] - l * branch[2]
                        reactive_equation += self.Q[j] - l * branch[3]
                        # 约束（6）
                        power_equation = 2 * (branch[2] * self.P[j] + branch[3] * self.Q[j]) - (branch[2] ** 2 + branch[3] ** 2) * l + self.v[i] - self.v[branch[0] - 1]
                        self.opti.subject_to(power_equation == 0)
                self.opti.subject_to(active_equation == 0)  # 约束（1）
                self.opti.subject_to(reactive_equation == 0)  # 约束（2）
            # 约束（7）
            self.opti.subject_to(self.v[i] >= bus[12])
            self.opti.subject_to(self.v[i] <= bus[11])

    def solve(self):
        self.objective()
        self.constraint()
        self.opti.solver('ipopt')
        self.sol = self.opti.solve()
        print(self.sol.value(self.p0))


OPF = DistributionSystem(1)
OPF.solve()
