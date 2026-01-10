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

#---------------------
logfile_name = 'logs/udp_log1.csv'
#-------------------
df = pd.read_csv(logfile_name)

plt.figure()
t_log = np.array(df['time'])
t_log -= t_log[0]
cycle = 1000.0*np.diff(t_log)

plt.subplot(211)
plt.hist(cycle,bins=100)
plt.xlabel('[ms]')
plt.ylabel('frequency')
plt.title(os.path.basename(__file__))

plt.subplot(212)
plt.plot(t_log[1:],cycle,'.-')
plt.legend(['Python cycle'])
plt.ylabel('[ms]')
plt.xlabel('time [s]')


plt.show()

