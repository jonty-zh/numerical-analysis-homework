import numpy as np
from scipy import interpolate

# 导入数据
years = np.arange(1960, 2021, 10)
populations = np.array([180671, 205052, 227225, 249623, 282162, 309327, 329484])

def _poly_newton_coefficient(x, y):
    m = len(x)
    x = np.copy(x)
    a = np.copy(y)
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k-1])/(x[k:m] - x[k - 1])
    return a

def newton_polynomial(x_data, y_data, x):
    '''Newton插值'''
    a = _poly_newton_coefficient(x_data, y_data)
    n = len(x_data) - 1  # Degree of polynomial
    p = a[n]
    for k in range(1, n + 1):
        p = a[n - k] + (x - x_data[n - k])*p
    return p

# Lagrange插值
f_l = interpolate.lagrange(years, populations)

# 三次样条插值
f_sp = interpolate.CubicSpline(years, populations)

test_years = [1950, 2005, 2030]
p_1950 = 151326
p_2005 = 295516

print("Lagrange插值结果")
for year in test_years:
    print("\t\t{}年: {}".format(year, f_l(year)))
er_l1 = (f_l(test_years[0])-p_1950)/p_1950
er_l2 = (f_l(test_years[1])-p_2005)/p_2005
print("\t{}年相对误差: {}".format(test_years[0], er_l1))
print("\t{}年相对误差: {}".format(test_years[1], er_l2))

print("Newton插值结果")
for year in test_years:
    print("\t\t{}年: {}".format(year, newton_polynomial(years, populations, year)))
er_n1 = (newton_polynomial(years, populations, test_years[0])-p_1950)/p_1950
er_n2 = (newton_polynomial(years, populations, test_years[1])-p_2005)/p_2005
print("\t{}年相对误差: {}".format(test_years[0], er_n1))
print("\t{}年相对误差: {}".format(test_years[1], er_n2))

print("三次样条插值结果")
for year in test_years:
    print("\t\t{}年: {}".format(year, f_sp(year)))
er_sp1 = (f_sp(test_years[0])-p_1950)/p_1950
er_sp2 = (f_sp(test_years[0])-p_2005)/p_2005
print("\t{}年相对误差: {}".format(test_years[0], er_sp1))
print("\t{}年相对误差: {}".format(test_years[1], er_sp2))
