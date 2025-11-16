import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

file_name = "logs/log.csv"

df = pd.read_csv(file_name)

plt.figure()
time = np.array(df['time'])
esp32_time0 = np.array(df['esp32_time0'])  
esp32_time1 = np.array(df['esp32_time1'])  
esp32_time2 = np.array(df['esp32_time2'])  
esp32_time3 = np.array(df['esp32_time3'])  

board_frame = np.array(df['board_frame'])

plt.subplot(211)
plt.plot(time[0:-1],np.diff(esp32_time0))
plt.ylim(-2,30)
plt.legend(['cycle time'])
plt.xlabel('time [s]')
plt.ylabel('[ms]')
plt.title(os.path.basename(__file__))

plt.subplot(212)
plt.plot(time,esp32_time2-esp32_time1,time,esp32_time1-esp32_time0,time,esp32_time3-esp32_time0)
plt.legend(['BNO055 read time','UDP send/recv','Thread start'])
plt.ylim(-5,20)
plt.ylabel('[ms]')
plt.xlabel('time [s]')

plt.show()

