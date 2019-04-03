import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sympy import Symbol, latex, diff


plt.xlabel('Temperature (°C)')
plt.ylabel('Conductivity (μS/cm)')

printFlagpoints = False
regression = True
uncertainties = True

def func(x, a, b, c):
    return a*x**2 + b*x + c

'''
Possible models:
    1. a/x
    2. a/x + b
    3. a/x + b*x + c
    4. a/x + b/x + c
    5. a/x + b*x**2 + c
    6. a/x + b/x**2 + c
    7. a*x**2 + b*x + c
'''

option = 'cacl2'

if option == 'nacl':
    data = pd.read_csv('nacl.csv')
elif option == 'kcl':
    data = pd.read_csv('kcl.csv')
elif option == 'cacl2':
    data = pd.read_csv('cacl2.csv')
else:
    data = pd.read_csv('mgcl2.csv')

if printFlagpoints:
    finalLength = len(data)
else:
    finalLength = 321

if option == 'mgcl2':
    time = data['Remote Data: Time (s)'][:finalLength]
    conductivity = data['Remote Data: Conductivity (µS/cm)'][:finalLength]
    temperature = data['Remote Data: Temperature (°C)'][:finalLength]
else:
    time = data['Run 1: Time (s)'][:finalLength]
    conductivity = data['Run 1: Conductivity (µS/cm)'][:finalLength]
    temperature = data['Run 1: Temperature (°C)'][:finalLength]


flagpoints = [0, 100, 200, 300, 400, 500, 600]

if printFlagpoints:
    print("Time:Conductivity:Temperature")
    for elem in flagpoints:
        print("{}:{}:{}".format(time[elem], conductivity[elem], temperature[elem]))

if uncertainties:
    print("Uncertainties in masses:")
    masses = {
        'nacl': 14.6107,
        'kcl': 18.6378,
        'mgcl2': 23.8027,
        'cacl2': 27.745
    }

    print((0.005 / masses[option]) * 100)

    print("*" * 50)
    print("Uncertainties in volumes:")
    vol = 250
    u = 0.5

    print((u / vol) * 100)

    print("*" * 50)
    print("Uncertainties in temperatures:")

    u_temps = []
    for elem in temperature:
        u_temps.append((0.005 / elem) * 100)

    print(sum(u_temps) / len(u_temps))


    print("*" * 50)
    print("Uncertainties in conductivities:")

    u_conds = []
    for elem in conductivity:
        u_conds.append((0.005 / elem) * 100)

    print(sum(u_conds) / len(u_conds))

if regression:
    x = np.array(temperature, dtype=float)
    y = np.array(conductivity, dtype=float)

    popt1, pcov1 = curve_fit(func, x, y)
    print("params = {}, R^2 = {}".format(popt1, r2_score(y, func(x, *popt1))))

    x_s = Symbol("x")
    model = popt1[0]*x_s**2 + popt1[1]*x_s + popt1[2]
    print("Model: {}".format(latex(model)))
    print("Derivative: {}".format(latex(diff(model))))

    plt.plot(x, func(x, *popt1), 'r-', label='Curve of best fit for {}'.format(option))
    plt.scatter(x, y, label='Raw data for {}'.format(option))
else:
    plt.plot(temperature, conductivity)

plt.legend(loc='upper right')
plt.show()

