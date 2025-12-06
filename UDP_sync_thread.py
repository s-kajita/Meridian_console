import sys
import os
import threading
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

MAX_LOG_SIZE = 2000
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

UDP_sync_finished = False

TAU_cycle = 0.01   # メインループサイクル [s]
tau_udp = 0.0025   # 目標受信タイミング [s]
tau_window = 0.005  # UDP受信ウィンドウのサイズ [s]

def udp_sync():
    global UDP_sync_finished, tau_udp, tau_window

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP用のsocket設定
    sock.bind((UDP_RECV_IP_DEF, UDP_RECV_PORT))


    # 180個の要素を持つint8型のNumPy配列を作成
    _r_bin_data = np.zeros(180, dtype=np.int8)

    sock.settimeout(0)  # 非ブロッキングモード

    tau_avg = 0
    tau_rcv = 0
    NoUDP = 0
    TotalN = 1500
    #TotalN = 5000
    print(f"Receiving UDP for {TotalN/100} s")
    for n in range(TotalN):
        Tcycle = time.perf_counter()    #　サイクル開始時刻
        Nrcv = 0
        tau = 0.0
        while tau < tau_window:
            tau = time.perf_counter() - Tcycle
            try:
                _r_bin_data, addr = sock.recvfrom(MSG_BUFF)
            except socket.error as e:
                pass
            else:
                Nrcv += 1           # at successful UDP receive
                tau_rcv = tau

        if Nrcv > 0:
            tau_avg += 0.01*(-tau_avg + tau_rcv)  # smoothing
        else:
            NoUDP += 1  # count 

        time_log.append(Tcycle)
        Nrcv_log.append(Nrcv)
        tau_log.append(tau_rcv)
        tau_avg_log.append(tau_avg)
        r_meridim_ushort = struct.unpack('90H', _r_bin_data)  # unsignedshort型
        esp32_time = [r_meridim_ushort[idx] for idx in [80,81,82,83]]
        esp32_time_log.append(esp32_time)

        # サイクル・受信タイミング制御
        if n < 300:
            tau_ctrl = 0.0
        else:
            tau_ctrl = 0.005*(tau_avg - tau_udp)    # UDP receive timing control

        while tau < TAU_cycle+tau_ctrl:
            tau = time.perf_counter() - Tcycle

    sock.close()
    print(f"{NoUDP} failed UDP receive out of {TotalN} attempts, {(NoUDP/TotalN)*100:.1f} %")

    UDP_sync_finished = True

def save_log(logfile_name):
    #------------ save log ---------
    print(f"Save {logfile_name}")
    f=open(logfile_name, 'w', newline='')
    writer=csv.writer(f)
    labels = ['time','Nrcv','tau','tau_avg']+['esp32_time'+str(i) for i in [0,1,2,3]]
    writer.writerow(labels)
    for i in range(len(time_log)):
        logdata = [time_log[i],Nrcv_log[i],tau_log[i],tau_avg_log[i]]+esp32_time_log[i]
        writer.writerow(logdata)
    f.close()


def plot_log(logfile_name):
    df = pd.read_csv(logfile_name)
    t_log = np.array(df['time'])
    t_log -= t_log[0]

    #----------- plot ESP32 log -----------
    plt.figure()
    esp32_t = np.array(df['esp32_time0']) 
    esp32_t -= esp32_t[0]
    esp32_cycle = np.diff(esp32_t)

    plt.subplot(211)
    plt.title(os.path.basename(__file__))
    plt.plot(t_log, df['esp32_time0'])
    plt.legend(['esp32_time0'])
    plt.xlabel('time [s]')
    '''
    plt.hist(esp32_cycle,bins=100)
    plt.xlim(0,50)
    plt.xlabel('[ms]')
    plt.ylabel('frequency')
    '''

    plt.subplot(212)
    plt.plot(t_log[1:],np.diff(esp32_t))
    plt.legend(['ESP32 cycle'])
    plt.ylim(0,50)
    plt.ylabel('[ms]')
    plt.xlabel('time [s]')

    #------------ plot UDP log ------
    plt.figure()
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
    plt.ylim(0,12)
    plt.ylabel('[ms]')
    plt.xlabel('time [s]')
    plt.title(os.path.basename(__file__))

    plt.subplot(312)
    plt.plot(t_log,df['Nrcv'],'.-')
    plt.ylabel('Nrcv')
    plt.xlabel('time [s]')

    plt.subplot(313)
    plt.plot(t_log,1000*df['tau'],'.-',t_log,1000*df['tau_avg'],t_log[[0,-1]],[tau_udp*1000,tau_udp*1000],'r--')
    plt.legend(['tau','tau_avg','tau_d'])
    plt.ylim(0,12)
    plt.ylabel('[ms]')
    plt.xlabel('time [s]')

    plt.show()


def main():
    global UDP_sync_finished, tau_udp

    t0 = time.time()
    while True:
        print(f"time= {(time.time()-t0):.1f}")
        time.sleep(1.0)
        if UDP_sync_finished:
            break

    logfile_name = 'logs/udp_sync.csv'
    save_log(logfile_name)
    plot_log(logfile_name)

# ================================================================================================================
# ---- スレッド処理 ------------------------------------------------------------------------------------------------
# ================================================================================================================
if __name__ == '__main__':  # スレッド2つで送受信と画面描写を並列処理
    thread1 = threading.Thread(target=udp_sync)  # サブスレッドでフラグ監視・通信処理・計算処理
    thread1.start()
    main()  # メインスレッド
