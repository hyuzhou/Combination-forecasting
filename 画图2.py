import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('C:/Users/dw/Desktop/2M_1step/NREL10min2m.csv')
data2 = pd.read_csv('C:/Users/dw/Desktop/5M_1step/NREL10min 5m.csv')

data = data['Avg Wind Speed @ 2m [m/s]']
data2 = data2['Avg Wind Speed @ 5m [m/s]']

data_train = data[:-144]
data_test = data[-144:]
data2_train = data2[:-144]
data2_test = data2[-144:]
fig = plt.figure(figsize=(8,6))

ax1 = fig.add_subplot(2,1,1)
plt.plot(data_train,linewidth=0.7)
plt.plot(data_test,color='r',linewidth=0.7)
plt.title('Data #1',fontsize = 16, family = 'Times New Roman')
plt.vlines(1008,0,14,color='black',linestyles='--',linewidth=0.7)
label =['Training-evaluation','Testing']
plt.legend(label,fontsize=8,loc=2)
plt.ylabel('Wind speed (m/s)',family='Times New Roman')
plt.xlim([0,1153])
plt.ylim([0,14])

ax2 = fig.add_subplot(2,1,2)
plt.plot(data2_train,linewidth=0.7)
plt.plot(data2_test,color='r',linewidth=0.7)
label2 =['Training-evaluation','Testing']

plt.legend(label2,fontsize=8,loc=2)
plt.vlines(1008,0,14,color='black',linestyles='--',linewidth=0.7)
# plt.yticks(np.arange(0,20,5))
plt.xlabel('Time(10min)',family='Times New Roman',fontsize=13)
plt.ylabel('Wind speed (m/s)',family='Times New Roman')
plt.xlim([0,1153])
plt.ylim([0,14])
plt.title('Data #2',fontsize = 16, family = 'Times New Roman')
plt.show()