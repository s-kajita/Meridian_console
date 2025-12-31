import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

file_name = "logs/log.csv"

df = pd.read_csv(file_name)
Trecv = df['Trecv']
Trecv_failed = (Trecv == 0.0)
Failed_percent = (sum(Trecv_failed)/len(Trecv))*100
print(f"UDP receive failed: {Failed_percent:3.1f} %" )

plt.figure()

time = np.array(df['time'])
cycle = 1000.0*np.diff(time)
hist_range = (0,max(15,max(cycle)))

plt.subplot(211)
plt.hist(cycle,range=hist_range,bins=100)
plt.xlabel('[ms]')
plt.ylabel('frequency')
plt.title(os.path.basename(__file__)+f" / UDP receive failed: {Failed_percent:3.1f} %")

plt.subplot(212)
plt.plot(time[0:-1],cycle,'.-',time,1000.0*df['Trecv'],'r')
plt.legend(['Tcycle','Trecv'])
plt.ylabel('[ms]')
plt.xlabel('time [s]')

plt.show()

