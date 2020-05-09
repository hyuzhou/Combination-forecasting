import pandas as pd
import numpy as np

dta = pd.read_csv('C:/Users/dw/Desktop/NREL.csv')

dta1min_2m = dta['Avg Wind Speed @ 2m [m/s]'].tolist()
dta1min_5m = dta['Avg Wind Speed @ 5m [m/s]'].tolist()
dta1min_10m = dta['Avg Wind Speed @ 10m [m/s]'].tolist()

dta10min_2m,dta10min_5m,dta10min_10m = [],[],[]


for i in np.arange(0,len(dta1min_2m),10):
    aver = (dta1min_2m[i]+dta1min_2m[i+1]+dta1min_2m[i+2]+dta1min_2m[i+3]+
            dta1min_2m[i+4]+dta1min_2m[i+5]+dta1min_2m[i+6]+dta1min_2m[i+7]+
            dta1min_2m[i+8]+dta1min_2m[i+9])/10
    dta10min_2m.append(aver)
dta10min_2m = np.array(dta10min_2m)

np.savetxt('C:/Users/dw/Desktop/temp1.csv',dta10min_2m)

for i in np.arange(0,len(dta1min_5m),10):
    aver = (dta1min_5m[i]+dta1min_5m[i+1]+dta1min_5m[i+2]+dta1min_5m[i+3]+
            dta1min_5m[i+4]+dta1min_5m[i+5]+dta1min_5m[i+6]+dta1min_5m[i+7]+
            dta1min_5m[i+8]+dta1min_5m[i+9])/10
    dta10min_5m.append(aver)
dta10min_5m = np.array(dta10min_5m)
np.savetxt('C:/Users/dw/Desktop/temp2.csv',dta10min_5m)

for i in np.arange(0,len(dta1min_10m),10):
    aver = (dta1min_10m[i]+dta1min_10m[i+1]+dta1min_10m[i+2]+dta1min_10m[i+3]+
            dta1min_10m[i+4]+dta1min_10m[i+5]+dta1min_10m[i+6]+dta1min_10m[i+7]+
            dta1min_10m[i+8]+dta1min_10m[i+9])/10
    dta10min_10m.append(aver)
dta10min_10m = np.array(dta10min_10m)
np.savetxt('C:/Users/dw/Desktop/temp3.csv',dta10min_10m)