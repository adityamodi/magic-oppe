import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt('out_ModelFail_MSE.txt')
error = np.loadtxt('out_ModelFail_MSE_error.txt')

leg = ['AM', 'DR', 'WDR', 'MAGIC']
x = [2]
for i in range(1,15):
	x.append(3+2**i)

plt.errorbar(x, data[:,0], error[:,0], capsize=3, marker='.', color='k')
plt.errorbar(x, data[:,5], error[:,5], capsize=3, marker='.', color='b')
plt.errorbar(x, data[:,6], error[:,6], capsize=3, marker='.', color='g')
plt.errorbar(x, data[:,7], error[:,7], capsize=3, marker='.', color='r')
plt.legend(leg,ncol=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of episodes')
plt.ylabel('MSE')
plt.show()