import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from keras import Sequential
from keras.layers import Dense,Activation

dta = pd.read_csv('C:/Users/dw/Desktop/2M_1step/NREL10min2m.csv')
# 共1152个数据，前1008个用作cross_validation (七折)每144个数据一折
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

# 5 inputs
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

# BPNN
def make_model():
    model = Sequential()
    model.add(Dense(10, input_dim = 6, init="uniform",activation="relu"))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

model = make_model()

runtimes = 20
pred_sum = np.zeros((144,1))
for j in range(runtimes):
    model.fit(train_x,train_y,nb_epoch = 1500, verbose = 0, batch_size = 64)
    print('=========run %d times=========='%((j+1)))
    pred_scale = model.predict(test_x)
    pred_ = (pred_scale * X.std(axis=0)[-1]) + X.mean(axis=0)[-1]
    pred_sum += pred_
pred_aver = pred_sum / runtimes

np.savetxt('C:/Users/dw/Desktop/temp.csv',pred_aver)