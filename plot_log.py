import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_name = "log.csv"

df = pd.read_csv(file_name)

plt.figure()
plt.plot(df['time'],df[['q8','q9','q10']])
plt.plot(df['time'],df[['qd8','qd9','qd10']],'--')
plt.xlabel('time [s]')
plt.ylabel('[deg]')
plt.legend(['q8','q9','q10','qd8','qd9','qd10'])
plt.show()

