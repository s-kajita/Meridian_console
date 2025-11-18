import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

file_name = "logs/log.csv"

df = pd.read_csv(file_name)

time = np.array(df['time'])
acc_labels = ['acc_x','acc_y','acc_z']
acc = np.array(df[acc_labels])  
gyro_labels = ['gyro_x','gyro_y','gyro_z']
gyro = np.array(df[gyro_labels])
rpy_labels = ['roll','pitch','yaw']
rpy = np.array(df[rpy_labels])

d_rpy = np.diff(rpy,axis=0)/0.01

for k in range(3):
    plt.figure()
    plt.subplot(211)
    plt.plot(time,rpy[:,k])
    plt.legend([rpy_labels[k]])
    #plt.ylim(-20,20)
    plt.ylabel('[deg]')
    #plt.xlabel('time [s]')
    plt.title(os.path.basename(__file__))

    plt.subplot(212)
    plt.plot(time[0:-1],d_rpy[:,k],time,gyro[:,k])
    plt.legend([f"d/dt({rpy_labels[k]})",gyro_labels[k]])
    plt.ylabel('[deg/s]')
    plt.xlabel('time [s]')

plt.show()


