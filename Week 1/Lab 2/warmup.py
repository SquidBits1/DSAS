from sklearn.datasets import load_sample_image
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
# I = load_sample_image('flower.jpg')
# print('image shape:', I.shape)
# plt.imshow(I[:,:,0],cmap='gray') # greyscale?
# plt.show()

T = pd.read_csv(r'C:\Users\gilad\Desktop\UNI\Year 2\DSAS\Lab1\titanic.csv')


T_sub = T.loc[:,["Age","Fare","Pclass"]]
print(T_sub.head(8))

T_sub = T_sub.to_numpy()

R = np.corrcoef(T_sub.T)

plt.figure()
plt.imshow(np.abs(R))
plt.colorbar()
plt.show()