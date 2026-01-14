import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

file_name = "logs/log.csv"

df = pd.read_csv(file_name)

plt.figure()
Tcycle = np.array(df['Tcycle'])
cycle = 1000.0*np.diff(Tcycle)   # [ms]

board_frame = np.array(df['board_frame'])
pc_frame = np.array(df['pc_frame'])

board_frame -= board_frame[0]
pc_frame -= pc_frame[0]

plt.subplot(211)
plt.plot(Tcycle, board_frame)
plt.plot(Tcycle, pc_frame)
plt.legend(['esp32 frame','pc frame'])
plt.xlabel('time[s]')
plt.ylabel('[steps]')
plt.title(os.path.basename(__file__))

plt.subplot(212)
plt.plot(Tcycle[0:-1],np.diff(board_frame))
plt.plot(Tcycle[0:-1],np.diff(pc_frame))
plt.legend(['esp32 frame','pc frame'])
plt.ylabel('[step]')
plt.xlabel('time [s]')

plt.show()

