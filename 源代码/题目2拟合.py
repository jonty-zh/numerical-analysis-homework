import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import math
from symbol import parameters
from typing_extensions import dataclass_transform
from unittest import result
import pandas
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
from scipy import optimize

df = pd.read_excel("../相关资源/数据/题目2拟合.xlsx", sheet_name="Sheet1")
data_w = df["w"]
data_r = df["R"]

def f_ab(params, x, y):
    a, b = params
    residual = y - (b + a*x)
    return residual

def f_abc(params, x, y):
    a, b, c = params
    residual = y - (b + a*x + c*x**2)
    return residual

data_lnw, data_lnr = [], []
for i in range(len(data_w)):
    data_lnw.append(math.log(data_w[i]))
    data_lnr.append(math.log(data_r[i]))

data_lnw = np.array(data_lnw)
data_lnr = np.array(data_lnr)

# 拟合lnR = lnb + alnw，确定a，b
params_ab = [0, 0]
result1 = optimize.leastsq(f_ab, params_ab, (data_lnw, data_lnr))
a1, lnb1 = result1[0]
b1 = math.exp(lnb1)
# 计算平方误差
err_sq1 = 0
for i in range(len(data_lnr)):
    err_sq1 += (data_r[i]-math.exp(lnb1+a1*data_lnw[i]))**2
    # err_sq1 += (data_lnr[i]-(lnb1+a1*data_lnw[i]))**2
print("lnR = lnb + alnw")
print("参    数: a={},b={}".format(a1, b1))
print("拟合方程: lnR = ln({}) + ({})lnw".format(b1, a1))
print("平方误差: {}\n".format(err_sq1))

# 拟合lnR = lnb + alnw + c(lnw)^2，确定a，b，c
params_abc = [0, 0, 0]
result = optimize.leastsq(f_abc, params_abc, (data_lnw, data_lnr))
a2, lnb2, c2 = result[0]
b2 = math.exp(lnb2)
# 计算平方误差
err_sq2 = 0
for i in range(len(data_lnr)):
    err_sq2 += (data_r[i]-math.exp(lnb2+a2*data_lnw[i]+c2*data_lnw[i]**2))**2
    # err_sq2 += (data_lnr[i]-(lnb1+a1*data_lnw[i]))**2
print("lnR = lnb + alnw + c(lnw)^2")
print("参    数: a={},b={},c={}".format(a2, b2, c2))
print("拟合方程: lnR = ln({}) + ({})lnw + ({})(lnw)^2".format(b2, a2, c2))
print("平方误差: {}\n".format(err_sq2))
