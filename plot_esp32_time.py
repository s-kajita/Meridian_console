import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

file_name = "logs/log.csv"

df = pd.read_csv(file_name)

plt.figure()
time = np.array(df['time'])
esp32_time = np.array(df['esp32_time'])  
board_frame = np.array(df['board_frame'])

plt.subplot(211)
plt.plot(time[0:-1],np.diff(board_frame))
plt.xlabel('time [s]')
plt.ylabel('increment')
plt.title(os.path.basename(__file__))

plt.subplot(212)
plt.plot(time[0:-1],np.diff(esp32_time))
plt.ylim(-2,50)
plt.ylabel('[ms]')
plt.xlabel('time [s]')

plt.show()

