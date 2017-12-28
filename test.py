import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm
mat = sio.loadmat('ex6data1.mat')
X = mat['X']
y = mat['y']
y = y.astype(int)
y=np.squeeze(y)
clf = svm.SVC(kernel='linear', C=10)
clf.fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none')
plt.show()
# plt.scatter(x[:,0],x[:,1],c=y,cmap='PuOr')
# plt.show()