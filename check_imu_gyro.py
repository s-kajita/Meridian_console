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

d_rpy = np.diff(rpy,axis=0)/0.01

plt.subplot(311)
plt.plot(time,gyro)
plt.legend(gyro_labels)
#plt.ylim(-20,20)
plt.ylabel('[?]')
plt.xlabel('time [s]')
plt.title(os.path.basename(__file__))

plt.subplot(312)
plt.plot(time[0:-1],d_rpy)
plt.legend(['droll','dpitch','dyaw'])
plt.ylabel('[deg/s]')
plt.xlabel('time [s]')

plt.subplot(313)
plt.plot(time[0:-1],d_rpy[:,1],time,gyro[:,1])
plt.legend(['dpitch','gyro_y'])
plt.ylabel('[deg/s]')
plt.xlabel('time [s]')


plt.show()

