import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from SVM import *
mat = sio.loadmat('ex6data1.mat')
x = mat['X']
y = mat['y']
y = y.astype(int)
y=np.squeeze(y)
plt.scatter(x[:,0],x[:,1],c=y,cmap='PuOr')
S = SVM(x,y,1)
# print S.info()
# print S.Xdata
# print np.dot(S.Ydata,S.Xdata)
S.train()
print S.Ydata
S.result_print()
# print S.Ydata.shape
# print S.alphas.shape
print S.E
# print S.Ydata
xx = np.linspace(1., 4., 6)
yy = (np.linspace(S.b,S.b,6)-S.w[0]*xx)/S.w[1]
plt.plot(xx,yy)
plt.show()