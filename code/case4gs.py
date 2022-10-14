from casadi import *

opti = Opti()
P12=opti.variable()
P13=opti.variable()
P24=opti.variable()
P34=opti.variable()

Q12=opti.variable()
Q13=opti.variable()
Q24=opti.variable()
Q34=opti.variable()

v1=opti.variable()
v2=opti.variable()
v3=opti.variable()
v4=opti.variable()

target=P12+P13+50*1000
opti.minimize(target)
opti.subject_to(v1>=0.9*230)
opti.subject_to(v1<=1.1*230)
opti.subject_to(v2>=0.9*230)
opti.subject_to(v2<=1.1*230)
opti.subject_to(v3>=0.9*230)
opti.subject_to(v3<=1.1*230)
opti.subject_to(v4>=0.9*230)
opti.subject_to(v4<=1.1*230)

I12=(P12**2+Q12**2)/v1**2
opti.subject_to(P12-I12*0.01008-P24-170*1000==0)
opti.subject_to(Q12-I12*0.0504-Q24-105.35*1000==0)

I13=(P13**2+Q13**2)/v1**2
opti.subject_to(P13-I13*0.00744-P34-200*1000==0)
opti.subject_to(Q13-I13*0.0372-Q34-123.94*1000==0)

I24=(P24**2+Q24**2)/v2**2
I34=(P34**2+Q34**2)/v3**2
opti.subject_to(P24-I24*0.00744+P34-I34*0.01272-80*1000==0)
opti.subject_to(Q24-I24*0.0372+Q34-I34*0.0636-49.58*1000==0)

opti.solver('ipopt')
sol = opti.solve()
print(sol.value(target))
