import sys
import os
import numpy as np
import time
import socket
from contextlib import closing
import struct
import csv
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque    
Tcycle_log = deque([])
Nrcv_log = deque([]) 
tau_log  = deque([])
tau_avg_log = deque([])

#---------------------
logfile_name = 'udp_log.csv'
#-------------------
df0 = pd.read_csv('logs/'+logfile_name)

Tudp = np.array(df0['Tudp'])
Tudp -= Tudp[0]
cycle0 = 1000.0*np.diff(Tudp)


plt.figure()
plt.subplot(211)
plt.hist(cycle0,bins=100)
plt.xlabel('[ms]')
plt.ylabel('frequency')
plt.title(os.path.basename(__file__)+f" [{logfile_name}]")

plt.subplot(212)
plt.plot(Tudp[1:],cycle0,'.-')
plt.legend(['Python cycle'])
plt.ylabel('[ms]')
plt.xlabel('time [s]')

#plt.show()

#------------------------------

tau_avg = 0
NoUDP = 0
TotalN = 1500
tau_ctrl = 0     # 受信周期調整項
tau_d = 0.002   # 目標受信タイミング [s]
tau_end = 0.008  # UDP受信ウィンドウの終わり
print(f"Receiving UDP for {TotalN/100} s")

Tstart = time.perf_counter()
Tdisp = 1.0;
cnt = 0
for n in range(TotalN):
    Tcycle = Tnow = time.perf_counter()-Tstart    #　UDP受信時刻
    Nrcv = 0
    tau = 0.0
    while Tnow - Tcycle < 0.01+tau_ctrl:
        Tnow = time.perf_counter()-Tstart
        if Tnow - Tcycle < tau_end:
            if Tnow > Tudp[cnt]:
                received = True
                cnt = min(cnt+1, len(Tudp)-1)
            else:
                received = False

            if received:
                Nrcv += 1
                tau = Tnow - Tcycle
                tau_avg += 0.01*(-tau_avg + tau)  # smoothing

    if Nrcv == 0:
        NoUDP += 1  # count 

    if n > 300:
        tau_ctrl = 0.005*(tau_avg - tau_d)    # UDP receive timing control

    Tcycle_log.append(Tcycle)
    Nrcv_log.append(Nrcv)
    tau_log.append(tau)
    tau_avg_log.append(tau_avg)

    if cnt == len(Tudp)-1:
        break

    if Tnow >= Tdisp:
        print(f"time = {Tdisp}")
        Tdisp += 1.0

Failed_percent = (NoUDP/TotalN)*100
print(f"{NoUDP} failed UDP receive out of {TotalN} attempts, {Failed_percent:.1f} %")

#------------ save log ---------
logfile_name = 'logs/udp_sync_sim.csv'
print(f"Save {logfile_name}")
f=open(logfile_name, 'w', newline='')
writer=csv.writer(f)
labels = ['Tcycle','Nrcv','tau','tau_avg']
writer.writerow(labels)
for i in range(len(Tcycle_log)):
    logdata = [Tcycle_log[i],Nrcv_log[i],tau_log[i],tau_avg_log[i]]
    writer.writerow(logdata)
f.close()


df = pd.read_csv(logfile_name)
Tcycle = np.array(df['Tcycle'])
Tcycle -= Tcycle[0]

#------------ plot UDP log ------
'''
plt.figure()
cycle = 1000.0*np.diff(t_log)

plt.subplot(211)
plt.plot(t_log[1:],cycle,'.-',t_log,1000*df['tau'],'.-',t_log,1000*df['tau_avg'],t_log[[0,-1]],[tau_d*1000,tau_d*1000],'r--')
plt.legend(['Python cycle','tau','tau_avg','tau_d'])
plt.ylim(0,12)
plt.ylabel('[ms]')
plt.xlabel('time [s]')
plt.title(os.path.basename(__file__)+f" / UDP receive failed: {Failed_percent:3.1f} %")

plt.subplot(212)
plt.plot(t_log,df['Nrcv'],'.-')
plt.ylabel('Nrcv')
plt.xlabel('time [s]')
'''
#--------------------------
Tcycle = np.array(df['Tcycle'])
cycle = 1000.0*np.diff(Tcycle)
tau = np.array(df['tau'])
hist_range = (0,max(15,max(cycle)))

plt.figure()
plt.subplot(211)
plt.hist(cycle,range=hist_range,bins=100)
plt.hist(1000*tau,range=hist_range, bins=100)
plt.xlabel('[ms]')
plt.ylabel('frequency')
plt.title(os.path.basename(__file__)+f" / UDP receive failed: {Failed_percent:3.1f} %")

plt.subplot(212)
plt.plot(Tcycle[0:-1],cycle,'.-',Tcycle,1000.0*tau,'.',Tcycle,1000.0*df['tau_avg'],'r')
plt.legend(['Tcycle','tau','tau_avg'])
plt.ylabel('[ms]')
plt.xlabel('time [s]')




plt.show()

