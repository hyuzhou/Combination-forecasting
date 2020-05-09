import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

dta=pd.read_csv('C:/Users/dw/Desktop/2M_1step/NREL10min2m.csv')

y=dta['Avg Wind Speed @ 2m [m/s]'][6:1152]
x1,x2,x3,x4,x5,x6 = dta['x1'][6:1152],dta['x2'][6:1152],\
                    dta['x3'][6:1152],dta['x4'][6:1152],dta['x5'][6:1152],dta['x6'][6:1152]

# 4 inputs
# X = np.array(pd.concat((x1,x2,x3,x4,y),axis=1))
# X_scale = scale(X)
#
# train_x = X_scale[:1002,:4]
# test_x = X_scale[1002:,:4]
# train_y = X_scale[:1002,4]
# test_y = X_scale[1002:,4]

# # 5 inputs
# X = np.array(pd.concat((x1,x2,x3,x4,x5,y),axis=1))
# X_scale = scale(X)
#
# train_x = X_scale[:1002,:5]
# test_x = X_scale[1002:,:5]
# train_y = X_scale[:1002,5]
# test_y = X_scale[1002:,5]

# 6 inputs
X = np.array(pd.concat((x1,x2,x3,x4,x5,x6,y),axis=1))
X_scale = scale(X)

train_x = X_scale[:1002,:6]
test_x = X_scale[1002:,:6]
train_y = X_scale[:1002,6]
test_y = X_scale[1002:,6]

from sklearn.svm import SVR
model = SVR(kernel='linear')
pred_sum = np.zeros((144,1))
runtimes = 20

for j in range(runtimes):
    model.fit(train_x,train_y)
    print('=========run %d times=========='%(j+1))
    pred_scale = model.predict(test_x)
    pred_ = (pred_scale * X.std(axis=0)[-1]) + X.mean(axis=0)[-1]
    pred_sum += np.array(pred_).reshape((144,1))
pred_aver = pred_sum / runtimes
np.savetxt('C:/Users/dw/Desktop/temp.csv',pred_aver)