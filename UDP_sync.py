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

UDP_RECV_PORT = 22222                       # 受信ポート
UDP_SEND_PORT = 22224                       # 送信ポート
MSG_SIZE = 90                               # Meridim配列の長さ(デフォルトは90)
MSG_BUFF = MSG_SIZE * 2                     # Meridim配列のバイト長さ
# ------------ データロガー用変数 ---------

MAX_LOG_SIZE = 1000
from collections import deque     # dequeはリングバッファ
time_log = deque([],maxlen=MAX_LOG_SIZE)
Nrcv_log = deque([],maxlen=MAX_LOG_SIZE) 
tau_log  = deque([],maxlen=MAX_LOG_SIZE)
tau_avg_log = deque([],maxlen=MAX_LOG_SIZE)
esp32_time_log = deque([],maxlen=MAX_LOG_SIZE)

UDP_SEND_IP_DEF= '192.168.11.12'
UDP_RECV_IP_DEF= '192.168.11.3'
NETWORK_MODE = 0

UDP_SEND_IP = UDP_SEND_IP_DEF

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP用のsocket設定
sock.bind((UDP_RECV_IP_DEF, UDP_RECV_PORT))


print("Start.")

# 180個の要素を持つint8型のNumPy配列を作成
_r_bin_data = np.zeros(180, dtype=np.int8)

sock.settimeout(0)  # 非ブロッキングモード
#sock.settimeout(0.0003)

tau_avg = 0
for n in range(1000):
    Tcycle = Tnow = time.perf_counter()    #　UDP受信時刻
    Nrcv = 0
    tau = 0.0
    while Tnow - Tcycle < 0.01:
        received = True
        try:
            _r_bin_data, addr = sock.recvfrom(MSG_BUFF)
        #except socket.timeout:
        #    received = False
        except socket.error as e:
            received = False
        Tnow = time.perf_counter()
        if received:
            Nrcv += 1
            tau = Tnow - Tcycle
            tau_avg += 0.02*(-tau_avg + tau)

    time_log.append(Tcycle)
    Nrcv_log.append(Nrcv)
    tau_log.append(tau)
    tau_avg_log.append(tau_avg)
    r_meridim_ushort = struct.unpack('90H', _r_bin_data)  # unsignedshort型
    esp32_time = [r_meridim_ushort[idx] for idx in [80,81,82,83]]
    esp32_time_log.append(esp32_time)

'''
    while True:
        Tnow = time.perf_counter()
        if Tnow - T_recv > 0.01:
            break
'''
sock.close()
print("End.") 

#---------------------
logfile_name = 'logs/udp_sync.csv'
print(f"Save {logfile_name}")
f=open(logfile_name, 'w', newline='')
writer=csv.writer(f)
labels = ['time','Nrcv','tau','tau_avg']+['esp32_time'+str(i) for i in [0,1,2,3]]
writer.writerow(labels)
for i in range(len(time_log)):
    logdata = [time_log[i],Nrcv_log[i],tau_log[i],tau_avg_log[i]]+esp32_time_log[i]
    writer.writerow(logdata)
f.close()

#-------------------
df = pd.read_csv(logfile_name)

plt.figure()
t_log = np.array(df['time'])
t_log -= t_log[0]
cycle = 1000.0*np.diff(t_log)

'''
plt.subplot(311)
plt.hist(cycle,bins=100)
#plt.xlim(0,15)
plt.xlabel('[ms]')
plt.ylabel('frequency')
plt.title(os.path.basename(__file__))
'''
plt.subplot(311)
plt.plot(t_log[1:],cycle,'.-')
plt.legend(['Python cycle'])
plt.ylim(0,15)
plt.ylabel('[ms]')
plt.xlabel('time [s]')
plt.title(os.path.basename(__file__))

plt.subplot(312)
plt.plot(t_log,df['Nrcv'],'.-')
plt.ylabel('Nrcv')
plt.xlabel('time [s]')

plt.subplot(313)
plt.plot(t_log,1000*df['tau'],'.-',t_log,1000*df['tau_avg'])
plt.legend(['tau','tau_avg'])
plt.ylabel('[ms]')
plt.xlabel('time [s]')

#----------------------
plt.figure()
esp32_t = np.array(df['esp32_time0']) 
esp32_t -= esp32_t[0]
esp32_cycle = np.diff(esp32_t)

plt.subplot(211)
plt.hist(esp32_cycle,bins=100)
plt.xlim(0,50)
plt.xlabel('[ms]')
plt.ylabel('frequency')
plt.title(os.path.basename(__file__))

plt.subplot(212)
plt.plot(esp32_t[1:]*0.001,np.diff(esp32_t))
plt.legend(['ESP32 cycle'])
plt.ylim(0,50)
plt.ylabel('[ms]')
plt.xlabel('time [s]')




plt.show()

