import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

useMeasuredTemp = False

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def func2(x, a, b):
    return a * np.exp(-b * x) + 27.7


option = 'cacl2'

if option == 'nacl':
    data = pd.read_csv('nacl.csv')
elif option == 'kcl':
    data = pd.read_csv('kcl.csv')
elif option == 'cacl2':
    data = pd.read_csv('cacl2.csv')
else:
    data = pd.read_csv('mgcl2.csv')


if option == 'mgcl2':
    time = data['Remote Data: Time (s)']  # [:321]
    conductivity = data['Remote Data: Conductivity (µS/cm)']
    temperature = data['Remote Data: Temperature (°C)']
else:
    time = data['Run 1: Time (s)']  # [:321]
    conductivity = data['Run 1: Conductivity (µS/cm)']
    temperature = data['Run 1: Temperature (°C)']

printFlagpoints = True
flagpoints = [0, 100, 200, 300, 400, 500, 600]


if printFlagpoints:
    print("Time:Conductivity:Temperature")
    for elem in flagpoints:
        print("{}:{}:{}".format(time[elem], conductivity[elem], temperature[elem]))

plt.plot(temperature, conductivity)

plt.legend(loc='upper right')
plt.show()

