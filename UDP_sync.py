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

MAX_LOG_SIZE = 5000
from collections import deque     # dequeはリングバッファ
Tcycle_log = deque([],maxlen=MAX_LOG_SIZE)
Nrcv_log = deque([],maxlen=MAX_LOG_SIZE) 
tau_log  = deque([],maxlen=MAX_LOG_SIZE)
tau_avg_log = deque([],maxlen=MAX_LOG_SIZE)
esp32_time_log = deque([],maxlen=MAX_LOG_SIZE)

#------ My room router -------
UDP_SEND_IP_DEF= '192.168.11.12'
UDP_RECV_IP_DEF= '192.168.11.3'

#------- ASUS WiFi router ----------
#UDP_SEND_IP_DEF= '192.168.50.145'
#UDP_RECV_IP_DEF= '192.168.50.142'

NETWORK_MODE = 0

UDP_SEND_IP = UDP_SEND_IP_DEF

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP用のsocket設定
sock.bind((UDP_RECV_IP_DEF, UDP_RECV_PORT))

# 180個の要素を持つint8型のNumPy配列を作成
_r_bin_data = np.zeros(180, dtype=np.int8)

sock.settimeout(0)  # 非ブロッキングモード

TAU_BUF_SIZE = 51
tau_buf = deque([],maxlen=TAU_BUF_SIZE)   #メディアンフィルタ用バッファ

tau_avg = 0
NoUDP = 0
tau_ctrl = 0     # 制御サイクル調整項
TAU_udp   = 0.001   # 目標受信タイミング [s]
TAU_WINDOW = 0.008  # UDP受信ウィンドウの長さ [s]

print(f"Receiving UDP for {MAX_LOG_SIZE/100} s")

#------------------ 制御サイクル ------------------
firsttime = True

Tstart = time.perf_counter()
Tdisp = 1.0;
for n in range(MAX_LOG_SIZE):
    Tcycle = time.perf_counter()    #　制御サイクル開始時刻
    Nrcv = 0
    
    if Tcycle - Tstart < 8.0:
        TAU_window = 0.0095    # 同期のため受信ウィンドウを広げる
    else:
        TAU_window = TAU_WINDOW  # 定常状態の受信ウィンドウ
    
    while True:
        tau = time.perf_counter() - Tcycle
        try:
            _r_bin_data, addr = sock.recvfrom(MSG_BUFF)  # UDP受信を試みる
        except socket.error as e:
            pass     # UDP受信失敗
        else:
            tau_udp = tau   #　UDP受信時刻
            Nrcv += 1   # サイクル内UDP受信回数
        if tau > TAU_window:
            break               #UDP受信の試みを終了

    if Nrcv > 0:
        tau_buf.append(tau_udp)
        # サイクル内平均受信タイミング
        if len(tau_buf) == TAU_BUF_SIZE:
            tau_sorted = np.sort(tau_buf)    # メディアンフィルタによる平滑化
            tau_q1 = tau_sorted[TAU_BUF_SIZE//4]  # 第1四分位のデータ (TAU_BUF_SIZE//2 ならメジアン)
            if firsttime:
                firsttime = False
                tau_avg = tau_q1
            else:
                tau_avg += 0.02*(-tau_avg + tau_q1)  # ローパスフィルタ
    else:
        tau_udp = 0      # 受信できなかった場合は0とする
        NoUDP += 1       # 受信ウィンドウ内でUDPが受信できなかった   

    if n > 200:
        tau_ctrl = 0.005*(tau_avg - TAU_udp)    # UDP receive timing control

    Tcycle_log.append(Tcycle)
    Nrcv_log.append(Nrcv)
    tau_log.append(tau_udp)
    tau_avg_log.append(tau_avg)
    r_meridim_ushort = struct.unpack('90H', _r_bin_data)  # unsignedshort型
    esp32_time = [r_meridim_ushort[idx] for idx in [80,81,82,83]]
    esp32_time_log.append(esp32_time)

    if Tcycle-Tstart >= Tdisp:
        print(f"time = {Tdisp}")
        Tdisp += 1.0

    # 制御サイクル調整
    while True:
        if time.perf_counter() - Tcycle > 0.01+tau_ctrl:
            break
    
       
sock.close()
print("Finished.") 
Failed_percent = (NoUDP/MAX_LOG_SIZE)*100
print(f"{NoUDP} failed UDP receive out of {MAX_LOG_SIZE} attempts, {Failed_percent:.1f} %")

#------------ save log ---------
logfile_name = 'logs/udp_sync.csv'
print(f"Save {logfile_name}")
f=open(logfile_name, 'w', newline='')
writer=csv.writer(f)
labels = ['Tcycle','Nrcv','tau','tau_avg']+['esp32_time'+str(i) for i in [0,1,2,3]]
writer.writerow(labels)
for i in range(len(Tcycle_log)):
    logdata = [Tcycle_log[i],Nrcv_log[i],tau_log[i],tau_avg_log[i]]+esp32_time_log[i]
    writer.writerow(logdata)
f.close()


df = pd.read_csv(logfile_name)
Tcycle = np.array(df['Tcycle'])
Tcycle -= Tcycle[0]
tau_udp = df['tau']

#----------- plot ESP32 log -----------
'''
plt.figure()
esp32_t = np.array(df['esp32_time0']) 
esp32_t -= esp32_t[0]
esp32_cycle = np.diff(esp32_t)

plt.subplot(211)
plt.title(os.path.basename(__file__))
plt.plot(Tcycle, df['esp32_time0'])
plt.legend(['esp32_time0'])
plt.xlabel('time [s]')

plt.subplot(212)
plt.plot(Tcycle[1:],np.diff(esp32_t))
plt.legend(['ESP32 cycle'])
plt.ylim(0,50)
plt.ylabel('[ms]')
plt.xlabel('time [s]')
'''

#------------ plot UDP log ------
'''
plt.figure()
cycle = 1000.0*np.diff(Tcycle)

plt.subplot(211)
plt.plot(Tcycle[1:],cycle,'.-',Tcycle,1000*df['tau'],'.-',Tcycle,1000*df['tau_avg'],Tcycle[[0,-1]],[tau_d*1000,tau_d*1000],'r--')
plt.legend(['Tcycle','tau','tau_avg','tau_d'])
plt.ylim(0,12)
plt.ylabel('[ms]')
plt.xlabel('time [s]')
plt.title(os.path.basename(__file__)+f" / UDP receive failed: {Failed_percent:3.1f} %")

plt.subplot(212)
plt.plot(Tcycle,df['Nrcv'],'.-')
plt.ylabel('Nrcv')
plt.xlabel('time [s]')

plt.show()
'''

plt.figure()

cycle = 1000.0*np.diff(Tcycle)
hist_range = (0,max(15,max(cycle)))

plt.subplot(211)
plt.hist(cycle,range=hist_range,bins=100)
plt.hist(1000*tau_udp,range=hist_range, bins=100)
plt.legend(['Tcycle','tau_udp'])
plt.xlabel('[ms]')
plt.ylabel('frequency')
plt.title(os.path.basename(__file__)+f" / UDP receive failed: {Failed_percent:3.1f} %")

plt.subplot(212)
plt.plot(Tcycle[0:-1],cycle,'.-',Tcycle,1000.0*tau_udp,'.',Tcycle,1000.0*df['tau_avg'],'r')
plt.legend(['Tcycle','tau_udp','tau_avg'])
plt.ylabel('[ms]')
plt.xlabel('time [s]')

plt.show()
