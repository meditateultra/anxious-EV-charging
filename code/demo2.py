from casadi import *

class fun:
    def __init__(self):
        self.opti = Opti()
        self.x=[self.opti.variable() for i in range(2)]

    def objective(self):
        self.f=0
        self.f+=(self.x[1]-self.x[0]**2)**2
        self.opti.minimize( self.f )

    def constraint(self):
        self.opti.subject_to( self.x[0]**2+self.x[1]**2==1 )
        self.opti.subject_to(       self.x[0]+self.x[1]>=1 )

    def solve(self):
        self.opti.solver('ipopt')
        sol = self.opti.solve()
        print(sol.value(self.x[0]))
        print(sol.value(self.x[1]))
        print(sol.value(self.f))
a=fun()
a.solve()