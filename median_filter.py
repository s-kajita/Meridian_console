import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import deque     # リングバッファ

#file_name = "udp2ms_window8ms.csv"
file_name = "log.csv"

df = pd.read_csv("logs/"+file_name)
Trecv = df['Trecv']
Trecv_failed = (Trecv == 0.0)
Failed_percent = (sum(Trecv_failed)/len(Trecv))*100
print(f"UDP receive failed: {Failed_percent:3.1f} %" )

time = np.array(df['time'])

TAU_BUF_SIZE = 41
tau_buf = deque([],maxlen=TAU_BUF_SIZE)

tau_median = 0
tau_avg = 0
tau_median_m = np.zeros(len(time))
tau_avg_m = np.zeros(len(time))
firsttime = True
for n in range(len(time)):
    tau_buf.append(Trecv[n])
    if len(tau_buf) == TAU_BUF_SIZE:
        tau_median = np.median(tau_buf)
        if firsttime:
            firsttime = False
            tau_avg = tau_median
        else:
            tau_avg += 0.01*(-tau_avg + tau_median)            
    tau_median_m[n]=tau_median
    tau_avg_m[n] = tau_avg

plt.figure()
plt.title(os.path.basename(__file__)+f" [{file_name}]  recv. failed: {Failed_percent:3.1f} %")
plt.plot(time,1000.0*df['Trecv'],'y.-',time,1000.0*df['tau_avg'],time,1000.0*tau_median_m,time,1000.0*tau_avg_m)
plt.legend(['Trecv','tau_avg','tau_median','tau_avg'])
plt.ylabel('[ms]')
plt.xlabel('time [s]')

plt.show()

