#Online Bayesian Linear Regression for minibatches
import numpy as np
from random import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
#Formatting content.......
def read_file(filename):

	"""
		Creating a list of lists containing the entire dataset
	"""
	with open(filename) as f:
	#Skip first 3 lines
		f.readline()
		f.readline()
		f.readline()

		
		content = f.readlines()
		content = ([x.strip() for x in content])
		data = []
		for i in range(0,len(content)):
			content[i] = content[i].split(' ')
			data.append(list(map(float,content[i])))
		data = np.array(data)
	return data

data = read_file('oakland_part3_am_rf.node_features')

labels = [1004, 1100, 1103, 1200, 1400]

def calc_label(y_o, i):
	y = y_o.copy()
	y[y == labels[i]] = 1
	# y[y == labels[j]] = 1
	y[y != 1] = 0
	return y
#---------------------------------------------------------Variables-----------------------------------------------------------------------
#Initialize hyperparameters
sigma = 0.1
#Initialize mean cov for distribution of weights
mu = np.zeros((10,1))
cov = 10*np.eye(10)

#Calculating natural parameters: J and P
P = LA.pinv(cov) #10x10 matrix
J = P.dot(mu) #10x1 matrix

num_class = 2 #Binary classification using BLR
num_pass = 10
num_batch = 10;
p = 0
loss = []
Weights = []  # Stores the best/average weight after each pass
#------------------------------------------------------------------------------------------------------------------------------------------
while(p<num_pass):# Multiple passes over the dataset --> to mimic online learning with continuous stream of data
	"""
	Shuffle data..		
	"""
	j = 0;
	np.random.shuffle(data)
	#Extract X and Y from the shuffled data
	y_original = data[:,4]
	x = data[:,5:]
	y = calc_label(y_original, 0).reshape(-1,1)
	
	predict = []
	while((j+num_batch) < len(data)):

		#Update J and P as new data comes in...
		
		#P = P +sum(xi *xi.T) --> 10x10 matrix... since xi is a row, calculating xi.T*xi
		#J = J +sum(xi.T*yi) --> 10x1 matrix........
		P = P + ((x[j:j+num_batch,:].T).dot(x[j:j+num_batch,:]))/(sigma**2)
		J = J + (x[j:j+num_batch,:].T.dot(y[j:j+num_batch]))/(sigma**2)

		#calculate new mean and cov from J and P
		cov_final = LA.pinv(P)
		mu_final = cov_final.dot(J)

		Predicted_mini = (mu_final.T).dot(x[j:j+num_batch,:].T).T
		#Predict..Appending prediction and true label for each data
		Predicted_mini[Predicted_mini>=0.5] = 1
		Predicted_mini[Predicted_mini!=1] = 0
		Concat_arr = np.concatenate((Predicted_mini, y[j:j+num_batch]),1)

		#Calculating Cross-Entropy loss for each minibatch
		train_loss = len(Concat_arr[Concat_arr[:,0]!=Concat_arr[:,1]])
		predict.extend(Concat_arr[:,0].tolist())		
		loss.append(train_loss)
		
		Weights.append((mu_final.T).tolist()[0])
		
		
		j = j+num_batch

	# if(j+num_batch != len(data)):
	# 	P = P + ((x[j:,:].T).dot(x[j:,:]))/(sigma**2)
	# 	J = J + (x[j:,:].T.dot(y[j:]))/(sigma**2)

	# 	#calculate new mean and cov from J and P
	# 	cov_final = LA.pinv(P)
	# 	mu_final = cov_final.dot(J)

	# 	Predicted_mini = (mu_final.T).dot(x[j:,:].T).T
	# 	#Predict..Appending prediction and true label for each data
	# 	Predicted_mini[Predicted_mini>=0.5] = 1
	# 	Predicted_mini[Predicted_mini!=1] = 0
	# 	Concat_arr = np.concatenate((Predicted_mini, y[j:]),1)

	# 	#Calculating Cross-Entropy loss for each minibatch
	# 	train_loss = len(Concat_arr[Concat_arr[:,0]!=Concat_arr[:,1]])
	# 	predict.extend(Concat_arr[:,0].tolist())		
	# 	loss.append(train_loss)
		
	# 	Weights.append((mu_final.T).tolist()[0])
	
	
	# print(sum(Weights))
	p = p+1

print(np.shape(np.array(Weights)))
best_wt = np.mean(np.array(Weights),0)


#Weights after training
print('Weights after training: {}'.format(best_wt))
print('Last Weight{}'.format(mu_final))
# print('Loss:{}'.format(loss))
w_t =np.array(best_wt)

#Try on testing data.......
test = read_file('oakland_part3_an_rf.node_features')
y_test = test[:,4]
y_test = calc_label(y_test,0).reshape(-1,1)
x_test = test[:,5:]
loss=[]
predict = []
p=0
t=0
#Pass test data also batch-wise
while(p<num_pass):
	while(t+num_batch < len(test)):
		Predicted_mini = (w_t.T).dot(x_test[t:t+num_batch,:].T).T
		#Predict..Appending prediction and true label for each data
		Predicted_mini[Predicted_mini>=0.5] = 1
		Predicted_mini[Predicted_mini!=1] = 0
		Predicted_mini = Predicted_mini.reshape(-1,1)
		
		Concat_test = np.concatenate((Predicted_mini, y_test[t:t+num_batch]),1)
		
		test_loss = len(Concat_test[Concat_test[:,0]!=Concat_test[:,1]])#No. of misclassified datapoints in each batch
		
		predict.extend(Concat_test.tolist())
		loss.append(test_loss)

		t = t+num_batch
	p = p+1

# print('Test_loss:{}'.format(loss))
i = np.arange(len(loss))*num_batch

predict = np.array(predict)

fig, ax = plt.subplots()
ax.plot(i, loss)
plt.show()

# for i in range(len(predict)):
# 	print(predict)

# labels = [0,1]
# #Plotting 3D cloud
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
# colors = ['green','brown']


# test = test[0:len(predict),:]
# print(np.shape(test))
# for index, label in enumerate(labels):
# 	print("l:{}".format(np.shape(predict[:,0]==label)))
# 	test_label = test[predict[:,0] == label]
# 	ax.scatter(test_label[:,0], test_label[:,1], test_label[:,2], c = colors[index])

# plt.show()