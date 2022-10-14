from casadi import *

opti = Opti()
v1=opti.variable()
v2=opti.variable()
P12=opti.variable()
Q12=opti.variable()

p0=P12**2+50
opti.minimize(p0)

# opti.subject_to(v1>=190)
# opti.subject_to(v1<=250)
# opti.subject_to(v2>=190)
# opti.subject_to(v2<=250)

I12=(P12**2)/v1**2
opti.subject_to(P12-I12*2-50==0)
# opti.subject_to(Q12-I12*1==30)
# opti.subject_to(v1-v2==2*(2*P12+1*Q12)-(4+1)*I12)

opti.solver('ipopt')
opti.solve()
# print(opti.debug.value())
print(opti.value(p0))