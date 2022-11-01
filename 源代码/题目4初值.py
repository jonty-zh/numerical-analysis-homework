import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt 

k = 6.22e-19
n1 = 2e3
n2 = 2e3
n3 = 3e3

def f(t,x):
    return k*(n1-x/2)**2*(n2-x/2)**2*(n3-3*x/4)**3 

result=integrate.solve_ivp(f,(0,0.2),[0], method='RK45', dense_output=True)
print(result)
print("解得在0.2s后将形成{}单位的氢氧化钾".format(result.sol(result.t)[0][-1]))
# t=np.linspace(0,0.2,101)
# plt.plot(t,result.sol(t)[0],label='numerical solution')
plt.plot(result.t,result.sol(result.t)[0],label='numerical solution')
plt.grid()
plt.legend()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.show()
