from sys import stdout

from pypower.api import runopf
from numpy import array, sum

from numpy import ones, zeros, r_, sort, exp, pi, diff, arange, real, imag

from numpy import flatnonzero as find

from pypower.idx_bus import BUS_I, PD, QD, GS, BS, BUS_AREA, VM, VA
from pypower.idx_gen import  QG, QMAX, QMIN, GEN_STATUS
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS

from pypower.isload import isload
from pypower.ppoption import ppoption


class MyCase:
    """case15da
       data from ...
       Das D, Kothari DP, Kalam A (1995) Simple and efficient method for load
       flow solution of radial distribution networks. Int J Electr Power
       Energy Syst 17:335-346. doi: 10.1016/0142-0615(95)00050-0
       URL: https://doi.org/10.1016/0142-0615(95)00050-0
        """
    data = {"version": '2'}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    data["baseMVA"] = 1.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    data["bus"] = array([
        [1, 3, 0, 0, 0, 0, 1, 1, 0, 11, 1, 1, 1],
        [2, 1, 0.0441, 0.044991, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [3, 1, 0.07, 0.0714143, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [4, 1, 0.14, 0.1428286, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [5, 1, 0.044, 0.044991, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [6, 1, 0.14, 0.1428286, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [7, 1, 0.14, 0.1428286, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [8, 1, 0.07, 0.0714143, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [9, 1, 0.07, 0.0714143, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [10, 1, 0.0441, 0.044991, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [11, 1, 0.14, 0.1428286, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [12, 1, 0.07, 0.0714143, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [13, 1, 0.0441, 0.044991, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [14, 1, 0.07, 0.0714143, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
        [15, 1, 0.14, 0.1428286, 0, 0, 1, 1, 0, 11, 1, 1.1, 0.9],
    ])

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    data["gen"] = array([
        [1,	0,	0,	10,	-10,	1,	100,	1,	10,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
    ])


    ## branch data
    # RateA不能为0，如果不限可以给高一点
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    data["branch"] = array(
        [[1, 2, 0.0112, 0.0109, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
         [2, 3, 0.0097, 0.0095, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
         [3, 4, 0.007, 0.0068, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
         [4, 5, 0.0126, 0.0085, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
         [2, 9, 0.0166, 0.0112, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
         [9, 10, 0.0139, 0.0094, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
         [2, 6, 0.0211, 0.0143, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
         [6, 7, 0.009, 0.0061, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
         [6, 8, 0.0103, 0.007, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
         [3, 11, 0.0148, 0.01, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
         [11, 12, 0.0202, 0.0136, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
         [12, 13, 0.0166, 0.0112, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
         [4, 14, 0.0184, 0.0124, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
         [4, 15, 0.0099, 0.0067, 0, 9900, 0, 0, 0, 0, 1, -360, 360]]
    )

    ##-----  OPF Data  -----##
    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    data["gencost"] = array([
        [2,	0,	0,	3,	0,	20,	0]
    ])
    # 输出发电机数据
    ppopt = ppoption(OUT_GEN=1)
    loadP=sum(data['bus'][:,2])
    loadQ=sum(data['bus'][:, 3])

    def getCase(self):
        return self.data

    def solve(self):
        result = runopf(self.data, self.ppopt)
        loss = self.calloss(result, self.ppopt)
        dist = {'loadP': self.loadP, 'loadQ': self.loadQ, 'lossP': loss[0], 'lossQ': loss[1]}
        print(dist)

    '''计算线路上的损失'''
    def calloss(self,baseMVA, bus=None, gen=None, branch=None, f=None, success=None,t=None, fd=None, ppopt=None):
        if isinstance(baseMVA, dict):
            have_results_struct = 1
            results = baseMVA
            if gen is None:
                ppopt = ppoption()  ## use default options
            else:
                ppopt = gen
            if (ppopt['OUT_ALL'] == 0):
                return  ## nothin' to see here, bail out now
            if bus is None:
                fd = stdout  ## print to stdout by default
            else:
                fd = bus
            baseMVA, bus, gen, branch, success, et = \
                results["baseMVA"], results["bus"], results["gen"], \
                results["branch"], results["success"], results["et"]
            if 'f' in results:
                f = results["f"]
            else:
                f = None
        else:
            have_results_struct = 0
            if ppopt is None:
                ppopt = ppoption()  ## use default options
                if fd is None:
                    fd = stdout  ## print to stdout by default
            if ppopt['OUT_ALL'] == 0:
                return  ## nothin' to see here, bail out now

        isOPF = f is not None  ## FALSE -> only simple PF data, TRUE -> OPF data

        ## options
        isDC = ppopt['PF_DC']  ## use DC formulation?
        OUT_ALL = ppopt['OUT_ALL']
        OUT_ANY = OUT_ALL == 1  ## set to true if any pretty output is to be generated
        OUT_SYS_SUM = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_SYS_SUM'])
        OUT_AREA_SUM = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_AREA_SUM'])
        OUT_BUS = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_BUS'])
        OUT_BRANCH = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_BRANCH'])
        OUT_GEN = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_GEN'])
        OUT_ANY = OUT_ANY | ((OUT_ALL == -1) and
                             (OUT_SYS_SUM or OUT_AREA_SUM or OUT_BUS or
                              OUT_BRANCH or OUT_GEN))

        if OUT_ALL == -1:
            OUT_ALL_LIM = ppopt['OUT_ALL_LIM']
        elif OUT_ALL == 1:
            OUT_ALL_LIM = 2
        else:
            OUT_ALL_LIM = 0

        OUT_ANY = OUT_ANY or (OUT_ALL_LIM >= 1)
        if OUT_ALL_LIM == -1:
            OUT_V_LIM = ppopt['OUT_V_LIM']
            OUT_LINE_LIM = ppopt['OUT_LINE_LIM']
            OUT_PG_LIM = ppopt['OUT_PG_LIM']
            OUT_QG_LIM = ppopt['OUT_QG_LIM']
        else:
            OUT_V_LIM = OUT_ALL_LIM
            OUT_LINE_LIM = OUT_ALL_LIM
            OUT_PG_LIM = OUT_ALL_LIM
            OUT_QG_LIM = OUT_ALL_LIM

        OUT_ANY = OUT_ANY or ((OUT_ALL_LIM == -1) and (OUT_V_LIM or OUT_LINE_LIM or OUT_PG_LIM or OUT_QG_LIM))
        ptol = 1e-4  ## tolerance for displaying shadow prices

        ## create map of external bus numbers to bus indices
        i2e = bus[:, BUS_I].astype(int)
        e2i = zeros(max(i2e) + 1, int)
        e2i[i2e] = arange(bus.shape[0])

        ## sizes of things
        nb = bus.shape[0]  ## number of buses
        nl = branch.shape[0]  ## number of branches
        ng = gen.shape[0]  ## number of generators

        ## zero out some data to make printout consistent for DC case
        if isDC:
            bus[:, r_[QD, BS]] = zeros((nb, 2))
            gen[:, r_[QG, QMAX, QMIN]] = zeros((ng, 3))
            branch[:, r_[BR_R, BR_B]] = zeros((nl, 2))

        ## parameters
        ties = find(bus[e2i[branch[:, F_BUS].astype(int)], BUS_AREA] !=
                    bus[e2i[branch[:, T_BUS].astype(int)], BUS_AREA])
        ## area inter-ties
        tap = ones(nl)  ## default tap ratio = 1 for lines
        xfmr = find(branch[:, TAP])  ## indices of transformers
        tap[xfmr] = branch[xfmr, TAP]  ## include transformer tap ratios
        tap = tap * exp(-1j * pi / 180 * branch[:, SHIFT])  ## add phase shifters
        nzld = find((bus[:, PD] != 0.0) | (bus[:, QD] != 0.0))
        sorted_areas = sort(bus[:, BUS_AREA])
        ## area numbers
        s_areas = sorted_areas[r_[1, find(diff(sorted_areas)) + 1]]
        nzsh = find((bus[:, GS] != 0.0) | (bus[:, BS] != 0.0))
        allg = find(~isload(gen))
        ong = find((gen[:, GEN_STATUS] > 0) & ~isload(gen))
        onld = find((gen[:, GEN_STATUS] > 0) & isload(gen))
        V = bus[:, VM] * exp(-1j * pi / 180 * bus[:, VA])
        out = find(branch[:, BR_STATUS] == 0)  ## out-of-service branches
        nout = len(out)
        if isDC:
            loss = zeros(nl)
        else:
            loss = baseMVA * abs(V[e2i[branch[:, F_BUS].astype(int)]] / tap -
                                 V[e2i[branch[:, T_BUS].astype(int)]]) ** 2 / \
                   (branch[:, BR_R] - 1j * branch[:, BR_X])

        fchg = abs(V[e2i[branch[:, F_BUS].astype(int)]] / tap) ** 2 * branch[:, BR_B] * baseMVA / 2
        tchg = abs(V[e2i[branch[:, T_BUS].astype(int)]]) ** 2 * branch[:, BR_B] * baseMVA / 2
        loss[out] = zeros(nout)
        fchg[out] = zeros(nout)
        tchg[out] = zeros(nout)
        # print(sum(real(loss)), sum(imag(loss)))
        return sum(real(loss)), sum(imag(loss))

    def globalReward(self,bus,action):
        '''
        :param bus: 当前的EV连在第几个bus上
        :param action: EV采取的行动（充电），EV充电时action>0，放电时action<0
        :return: 返回全局reward
        '''
        self.data['bus'][bus]+=action
        result = runopf(self.data, self.ppopt)
        loss = self.calloss(result, self.ppopt)
        # EV的action带给变电站的功耗变化就是action加上线路上的损失（不对）
        return -(action+loss[0])


mycase = MyCase()
mycase.solve()


