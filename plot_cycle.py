import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

file_name = "logs/log.csv"

df = pd.read_csv(file_name)

plt.figure()
time = np.array(df['time'])
cycle = 1000.0*np.diff(time)

plt.subplot(211)
plt.hist(cycle,bins=100)
plt.xlabel('[ms]')
plt.ylabel('frequency')
plt.title(os.path.basename(__file__))

plt.subplot(212)
plt.plot(time[0:-1],cycle,'.-',time,1000.0*df['Trecv'],'r')
plt.ylabel('[ms]')
plt.xlabel('time [s]')

plt.show()

