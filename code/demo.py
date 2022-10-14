# from casadi import *

# x = SX.sym('x'); y = SX.sym('y'); z = SX.sym('z')
# nlp = {'x':vertcat(x,y,z), 'f':x**2+100*z**2, 'g':z+(1-x)**2-y}
# S = nlpsol('S', 'ipopt', nlp)
# print(S)
#
# r = S(x0=[2.5,3.0,0.75], lbg=0, ubg=0)
# x_opt = r['x']
# print('x_opt: ', x_opt)
import casadi

opti = casadi.Opti()

x = opti.variable()
y = opti.variable()

f=(y-x**2)**2
opti.minimize( f )
opti.subject_to( x**2+y**2==1 )
opti.subject_to(       x+y>=1 )

opti.solver('ipopt')


sol = opti.solve()

print(sol.value(x))
print(sol.value(y))

print(sol.value(f))