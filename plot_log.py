import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
file_name = "logs/log.csv"

df = pd.read_csv(file_name)

plt.figure()
plt.plot(df['Tcycle'],df[['q8','q9','q10']])
plt.plot(df['Tcycle'],df[['qd8','qd9','qd10']],'--')
plt.xlabel('time [s]')
plt.ylabel('[deg]')
plt.legend(['q8','q9','q10','qd8','qd9','qd10'])
plt.title(os.path.basename(__file__))
plt.show()

