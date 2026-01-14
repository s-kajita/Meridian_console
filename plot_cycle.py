import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

file_name = "logs/log.csv"

df = pd.read_csv(file_name)
tau_udp = df['tau_udp']
tau_udp_failed = (tau_udp == 0.0)
Failed_percent = (sum(tau_udp_failed)/len(tau_udp))*100
print(f"UDP receive failed: {Failed_percent:3.1f} %" )

plt.figure()

Tcycle = np.array(df['Tcycle'])
cycle = 1000.0*np.diff(Tcycle)
hist_range = (0,max(15,max(cycle)))

plt.subplot(211)
plt.hist(cycle,range=hist_range,bins=100)
plt.hist(1000*tau_udp,range=hist_range, bins=100)
plt.xlabel('[ms]')
plt.ylabel('frequency')
plt.title(os.path.basename(__file__)+f" / UDP receive failed: {Failed_percent:3.1f} %")

plt.subplot(212)
plt.plot(Tcycle[0:-1],cycle,'.-',Tcycle,1000.0*tau_udp,'.',Tcycle,1000.0*df['tau_avg'],'r')
plt.legend(['Tcycle','tau_udp','tau_avg'])
plt.ylabel('[ms]')
plt.xlabel('time [s]')

plt.show()

