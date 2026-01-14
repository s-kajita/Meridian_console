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

MAX_LOG_SIZE = 1500
from collections import deque     # dequeはリングバッファ
Tudp_log = deque([],maxlen=MAX_LOG_SIZE)
esp32_time_log = deque([],maxlen=MAX_LOG_SIZE)


def check_valid_ip(ip):  # IPアドレスの書式確認
    parts = ip.split(".")
    return (
        len(parts) == 4 and
        all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)
    )



def select_network_mode_and_ip(filename="board_ip.txt"):
    import re
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    # デフォルト値
    default = {
        0: {'SEND': '192.168.3.45', 'RECV': '192.168.3.3'},   # WIFI(DHCP)
        1: {'SEND': '192.168.3.45', 'RECV': '192.168.3.3'},   # WIFI(Fixed)
        2: {'SEND': '192.168.90.1', 'RECV': '192.168.90.2'},  # 有線LAN
    }
    mode_labels = {0: 'WIFI(DHCP)', 1: 'WIFI(Fixed)', 2: 'Wired LAN'}
    config = {}
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                m = re.match(r'([A-Z_]+)\s*=\s*"?([^"]*)"?', line.strip())
                if m:
                    k, v = m.group(1), m.group(2)
                    config[k] = v
    try:
        network_mode = int(config.get('NETWORK_MODE', '0'))
    except Exception:
        network_mode = 0
    wifi_dhcp_send = config.get(
        'UDP_WIFI_DHCP_SEND_IP_DEF', default[0]['SEND'])
    wifi_dhcp_recv = config.get(
        'UDP_WIFI_DHCP_RECV_IP_DEF', default[0]['RECV'])
    wifi_fixed_send = config.get(
        'UDP_WIFI_FIXED_SEND_IP_DEF', default[1]['SEND'])
    wifi_fixed_recv = config.get(
        'UDP_WIFI_FIXED_RECV_IP_DEF', default[1]['RECV'])
    wired_send = config.get('UDP_WIRED_SEND_IP_DEF', default[2]['SEND'])
    wired_recv = config.get('UDP_WIRED_RECV_IP_DEF', default[2]['RECV'])
    ip_table = {
        0: {'SEND': wifi_dhcp_send, 'RECV': wifi_dhcp_recv},
        1: {'SEND': wifi_fixed_send, 'RECV': wifi_fixed_recv},
        2: {'SEND': wired_send, 'RECV': wired_recv},
    }
    while True:
        print(f"Use previous {mode_labels[network_mode]} settings?")
        print(
            f"SEND_IP: {ip_table[network_mode]['SEND']}, RECV_IP: {ip_table[network_mode]['RECV']}")
        yn = input("y/n (Enter for y): ").strip().lower()
        if yn in ['', 'y', 'yes']:
            break
        elif yn in ['n', 'no']:
            while True:
                print("Please select a mode. \n0:WIFI(DHCP), 1:WIFI(Fixed), 2:Wired LAN")
                mode_in = input(
                    f"Enter mode number (current: {network_mode}): ").strip()
                if mode_in == '':
                    break
                try:
                    mode_in = int(mode_in)
                    if mode_in in [0, 1, 2]:
                        network_mode = mode_in
                        break
                except Exception:
                    pass
                print("Please enter 0, 1, or 2.")
            for key in ['SEND', 'RECV']:
                label = f"Enter the {key} IP for {mode_labels[network_mode]} (current: {ip_table[network_mode][key]}):"
                while True:
                    ip_in = input(label).strip()
                    if ip_in == '':
                        break
                    if check_valid_ip(ip_in):
                        ip_table[network_mode][key] = ip_in
                        break
                    print("Invalid IP address format. Example: 192.168.1.100")
            if network_mode == 0:
                config['UDP_WIFI_DHCP_SEND_IP_DEF'] = ip_table[0]['SEND']
                config['UDP_WIFI_DHCP_RECV_IP_DEF'] = ip_table[0]['RECV']
            elif network_mode == 1:
                config['UDP_WIFI_FIXED_SEND_IP_DEF'] = ip_table[1]['SEND']
                config['UDP_WIFI_FIXED_RECV_IP_DEF'] = ip_table[1]['RECV']
            elif network_mode == 2:
                config['UDP_WIRED_SEND_IP_DEF'] = ip_table[2]['SEND']
                config['UDP_WIRED_RECV_IP_DEF'] = ip_table[2]['RECV']
            config['NETWORK_MODE'] = str(network_mode)
            with open(filepath, 'w') as f:
                f.write(
                    f'UDP_WIFI_DHCP_SEND_IP_DEF="{config.get("UDP_WIFI_DHCP_SEND_IP_DEF", default[0]["SEND"])}"\n')
                f.write(
                    f'UDP_WIFI_DHCP_RECV_IP_DEF="{config.get("UDP_WIFI_DHCP_RECV_IP_DEF", default[0]["RECV"])}"\n')
                f.write(
                    f'UDP_WIFI_FIXED_SEND_IP_DEF="{config.get("UDP_WIFI_FIXED_SEND_IP_DEF", default[1]["SEND"])}"\n')
                f.write(
                    f'UDP_WIFI_FIXED_RECV_IP_DEF="{config.get("UDP_WIFI_FIXED_RECV_IP_DEF", default[1]["RECV"])}"\n')
                f.write(
                    f'UDP_WIRED_SEND_IP_DEF="{config.get("UDP_WIRED_SEND_IP_DEF", default[2]["SEND"])}"\n')
                f.write(
                    f'UDP_WIRED_RECV_IP_DEF="{config.get("UDP_WIRED_RECV_IP_DEF", default[2]["RECV"])}"\n')
                f.write(f'NETWORK_MODE = {network_mode}\n')
            print("Settings saved.\n")
            break  # 設定保存後は即ループを抜けてサービス開始
        else:
            print("Please answer with y or n.")
    return ip_table[network_mode]['SEND'], ip_table[network_mode]['RECV'], network_mode


UDP_SEND_IP_DEF, UDP_RECV_IP_DEF, NETWORK_MODE = select_network_mode_and_ip()
UDP_SEND_IP = UDP_SEND_IP_DEF

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP用のsocket設定
sock.bind((UDP_RECV_IP_DEF, UDP_RECV_PORT))

print("Start.")

# 180個の要素を持つint8型のNumPy配列を作成
_r_bin_data_past = np.zeros(180, dtype=np.int8)
_r_bin_data = np.zeros(180, dtype=np.int8)

#sock.settimeout(0)  # 非ブロッキングモード
#sock.settimeout(0.0003)

with closing(sock):
    Tstart = time.perf_counter()
    Tdisp = 1.0;
    for n in range(500):
        _r_bin_data, addr = sock.recvfrom(MSG_BUFF)
        Tudp = time.perf_counter()    #　UDP受信時刻
        Tudp_log.append(Tudp)
        r_meridim_ushort = struct.unpack('90H', _r_bin_data)  # unsignedshort型
        esp32_time = [r_meridim_ushort[idx] for idx in [80,81,82,83]]
        esp32_time_log.append(esp32_time)
        if Tudp-Tstart >= Tdisp:
            print(f"time = {Tdisp}")
            Tdisp += 1.0

print("End.") 

#---------------------
logfile_name = 'logs/udp_log.csv'
print(f"Save {logfile_name}")
f=open(logfile_name, 'w', newline='')
writer=csv.writer(f)
labels = ['Tudp']+['esp32_time'+str(i) for i in [0,1,2,3]]
writer.writerow(labels)
for i in range(len(Tudp_log)):
    logdata = [Tudp_log[i]]+esp32_time_log[i]
    writer.writerow(logdata)
f.close()

#-------------------
df = pd.read_csv(logfile_name)

plt.figure()
Tudp = np.array(df['Tudp'])
Tudp -= Tudp[0]
cycle = 1000.0*np.diff(Tudp)

plt.subplot(211)
plt.hist(cycle,bins=100)
plt.xlabel('[ms]')
plt.ylabel('frequency')
plt.title(os.path.basename(__file__))

plt.subplot(212)
plt.plot(Tudp[1:],cycle,'.-')
plt.legend(['Tudp cycle'])
plt.ylabel('[ms]')
plt.xlabel('time [s]')

#----------------------
'''
plt.figure()
esp32_t = np.array(df['esp32_time0']) 
esp32_cycle = np.diff(esp32_t)

plt.subplot(211)
plt.hist(esp32_cycle,bins=100)
plt.xlabel('[ms]')
plt.ylabel('frequency')
plt.title(os.path.basename(__file__))

plt.subplot(212)
plt.plot(esp32_t[0:-1]*0.001,np.diff(esp32_t))
plt.legend(['ESP32 cycle'])
plt.ylim(-2,40)
plt.ylabel('[ms]')
plt.xlabel('time [s]')
'''



plt.show()

