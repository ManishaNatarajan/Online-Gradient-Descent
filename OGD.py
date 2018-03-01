#Online Gradient Descent
import numpy as np
from random import *
import matplotlib.pyplot as plt
with open('oakland_part3_am_rf.node_features') as f:
	f.readline()
	f.readline()
	f.readline()


	theta = np.random.uniform(-10, 10,(10,1))#Theta is column vector
	# print(theta)
	L = []
	alpha = 0.6
	content = f.readline()
	
	#while(content != ''):
	for i in range(0,2):	
		content = content.split()
		# print(content)
		# # content.remove('')
		data = np.array(map(float,content)).reshape(-1,1)
		# print(np.shape(data))
		x = data[5:,:]
		y = data[4,:]
		#print(np.shape(x))
		# print(np.shape(y))	
		# print(np.shape(theta.dot(x)))
		loss = 0.5*(np.transpose(theta).dot(x) -y)**2
		# L.append(loss[0,0])
		print('wtx{}'.format(np.transpose(theta).dot(x)))
		print('y{}'.format(y))
		print('loss{}'.format(loss))
		theta = theta - alpha*(np.transpose(theta)*x - y)*x
		content = f.readline()

	# print(np.shape(range(0,len(L))))
	# print(np.shape(L))
	#print(L)
	# plt.plot(range(0,len(L)),L)
	# plt.show()

