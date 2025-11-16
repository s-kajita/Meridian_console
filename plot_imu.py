import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

file_name = "logs/log.csv"

df = pd.read_csv(file_name)

plt.figure()
time = np.array(df['time'])
acc_labels = ['acc_x','acc_y','acc_z']
acc = np.array(df[acc_labels])  
gyro_labels = ['gyro_x','gyro_y','gyro_z']
gyro = np.array(df[gyro_labels])
rpy_labels = ['roll','pitch','yaw']
rpy = np.array(df[rpy_labels])

plt.subplot(311)
plt.plot(time,acc)
plt.legend(acc_labels)
plt.xlabel('time [s]')
plt.ylabel('[m/s^2]')
plt.title(os.path.basename(__file__))

plt.subplot(312)
plt.plot(time,gyro)
plt.legend(gyro_labels)
plt.ylim(-20,20)
plt.ylabel('[rad/s]')
plt.xlabel('time [s]')

plt.subplot(313)
plt.plot(time,rpy)
plt.legend(rpy_labels)
plt.ylabel('[deg]')
plt.xlabel('time [s]')



plt.show()

