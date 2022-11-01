import math
import numpy as np
from scipy.integrate import romberg

err = math.inf
err_limits = 1e-10
print("设置误差限为{}".format(err_limits))

print("Romberg求积")
xi = 0.5
round = 0
while err>err_limits:
    # 被积函数
    g = lambda x: 1/math.sqrt(2*math.pi)*math.exp(-x**2/2)
    # Romberg求积
    fri = romberg(g, 0, xi, show=False)
    f = fri - 0.45
    # Newton迭代中的f'(x)
    df = 1/math.sqrt(2*math.pi)*math.exp(-xi**2/2)
    # Newton迭代
    xii = xi - f/df
    # 计算两次求得得根之间的误差
    err = abs(xii - xi)
    xi = xii
    round += 1
    print("第{}次Newton迭代后，Romberg积分值为{}，\t根为{}".format(round, fri, xi))

print("\nGauss-Lgendre求积")
xi = 0.5
round = 0
err = math.inf
while err>err_limits:
    # 被积函数
    gl = lambda x: xi*math.exp(-xi*xi*(x+1)**2/8)/(math.sqrt(2*math.pi)*2)
    # Gauss-Legendre求积
    gli = 5/9*gl(-math.sqrt(3/5))+8/9*gl(0)+5/9*gl(math.sqrt(3/5))
    f = gli - 0.45
    # Newton迭代中的f'(x)
    df = 1/math.sqrt(2*math.pi)*math.exp(-xi**2/2)
    # Newton迭代
    xii = xi - f/df
    # 计算两次求得得根之间的误差
    err = abs(xii - xi)
    xi = xii
    round += 1
    print("第{}次Newton迭代后，Gauss-Legendre积分值为{}，\t根为{}".format(round, gli, xi))
