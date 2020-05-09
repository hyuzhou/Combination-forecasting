import pandas as pd
import numpy as np
data = pd.read_csv('C:/Users/dw/Desktop/2M_3step/Contribution.csv')

df = data.iloc[:,1:]
df_actual = data.iloc[:,0]
matrix = np.array(df)
matrix_actual = np.array(df_actual)

# calculate score of each model
for i in range(0,12):
    a = np.concatenate([matrix[:,0:i],matrix[:,(i+1):12]],axis=1).sum(1)/ 11
    score = (100 - (np.abs(a - matrix_actual)/ matrix_actual) * 100)
    # np.savetxt('C:/Users/dw/Desktop/temp.csv', score)
    score_average = score.sum(0)/1000
    print(score_average)

# calculate score of total
c = matrix.sum(1)/12
score_total = (100 - (np.abs(c - matrix_actual)/ matrix_actual) * 100)
score_total_average = score_total.sum(0)/1000
print(score_total_average)
# np.savetxt('C:/Users/dw/Desktop/temp.csv',score_total)

