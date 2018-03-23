#Online Gradient Descent
import numpy as np
from random import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#Formatting content.......

print("Reading data.............")
def read_file(filename):
    with open(filename) as f:
        f.readline()
        f.readline()
        f.readline()

        """
        Creating a list of lists containing the entire dataset
        """
        content = f.readlines()
        content = ([x.strip() for x in content])
        data = []
        for i in range(0,len(content)):
            content[i] = content[i].split(' ')
            data.append(list(map(float,content[i])))
        data = np.array(data)
    return data

data = read_file('oakland_part3_am_rf.node_features')
data[:,4] = data[:,4].astype(int)
test = read_file('oakland_part3_an_rf.node_features')
test[:,4] = test[:,4].astype(int)


###########################################################################################
print("Training.............")

labels = [1004, 1100, 1103, 1200, 1400]

def calc_label(y_o, i):
    y = y_o.copy()
    y[y == labels[i]] = 1
    y[y!= 1] = 0
    return y


#Create a vector of classifiers.. one for each class 
w_t = []
for i in range(5):
	w = []
	for j in range(10):
		w.append(uniform(0,1)*0.5)
	w_t.append(w)

w_t = np.array(w_t)


#Initialize classifiers, labels and step_size
print("Hyper parameters.............")
alpha = 0.01 #Step_size
num_class = len(labels)
num_pass = 100
p = 0
lamb = 0
batch_size = 50

print("alpha: {}".format(alpha))
print("num of passes: {}".format(num_pass))
print("lambda: {}".format(lamb))

label_conv = {0:1004, 1:1100, 2:1103, 3:1200, 4:1400}

#3D array to store the weights
w = np.zeros(((int(len(data)/batch_size) + 1)*num_pass,5,10))

cross_entropy_train = []
iteration_batch = 0
while(p < num_pass): # Infinite passes over the dataset --> to mimic online learning with continuous stream of data
    """
    Shuffle data..        
    """
    np.random.shuffle(data)
    #Extract X and Y from the shuffled data
    y_original = data[:,4].copy()
    x = data[:,5:]
    Y = np.zeros((len(data), num_class))

    for c in range(num_class):
        Y[:,c] = calc_label(y_original, c)

    predict_train = []
    index = 0
    
    while(index + batch_size < len(data)): #Passing each data one at at time.. like online learning with sequential data streaming
        

        x_batch = x[index: index+batch_size,:]
        y_batch = Y[index: index+batch_size,:]
        z = w_t.dot(x_batch.T).T

        softmax = np.exp(z)
        softmax = softmax/np.sum(softmax, axis = 1).reshape(-1,1)

        predict = np.argmax(softmax, axis =1)

        l = (-1/batch_size)*np.trace(y_batch.dot(np.log(softmax).T)) + 0.5*lamb*np.sum(w_t*w_t)

        w[iteration_batch,:,:] = w_t

        w_t = w_t - alpha*((1/batch_size)*(softmax - y_batch).T.dot(x_batch) + lamb*w_t)

        predict_train.extend(predict.tolist())

        cross_entropy_train.append(l)

        index += batch_size
        iteration_batch+=1

    if(index < len(data)): 
        bs = len(data) - index

        x_batch = x[index: index+bs,:]
        y_batch = Y[index: index+bs,:]
        z = w_t.dot(x_batch.T).T

        softmax = np.exp(z)
        softmax = softmax/np.sum(softmax, axis = 1).reshape(-1,1)

        predict = np.argmax(softmax, axis =1)

        l = (-1/bs)*np.trace(y_batch.dot(np.log(softmax).T))

        w[iteration_batch,:,:] = w_t

        w_t = w_t - alpha*(1/bs)*(softmax- y_batch).T.dot(x_batch)

        predict_train.extend(predict.tolist())

        cross_entropy_train.append(l)

        iteration_batch+=1


    
    print('current pass: {}'.format(p))
    p = p + 1

predict_train = [label_conv[x] for x in predict_train]
predict_train = np.array(predict_train)
print('Train_predict{}'.format(np.shape(predict_train)))

####################################################################################################################################
#Plotting training cross entropy pass wise
print('Plotting training cross entropy pass wise')
f,ax = plt.subplots()
ax.plot(range(iteration_batch), cross_entropy_train)
plt.title('Cross Entropy training error batch wise')
plt.xlabel('Batch Iterations')
plt.ylabel('Softmax cross entropy error')
plt.show()


#####################################################################################################################################
#TESTING
print("Testing Calculations.............")
#Try on testing data....... (Calculating test loss)
#Testing data
y_test = test[:,4]
x = test[:,5:]
Y_test = np.zeros((len(test), num_class))

for c in range(num_class):
    Y_test[:,c] = calc_label(y_test, c)

#Average weights 
w_avg = np.mean(w, axis = 0)
index = 0
predict_test = []
cross_entropy_test = []
while(index + batch_size < len(test)):

    x_batch = x[index: index+batch_size,    :]
    y_batch = Y_test[index: index+batch_size,:]

    z = w_avg.dot(x_batch.T).T
    softmax = np.exp(z)
    softmax = softmax/np.sum(softmax, axis =1).reshape(-1,1)
    predict = np.argmax(softmax, axis =1)
    l = (-1/batch_size)*np.trace(y_batch.dot(np.log(softmax).T))
    predict_test.extend(predict.tolist())
    cross_entropy_test.append(l)
    index+=batch_size

if(index < len(test)):
    bs = len(test) - index
    x_batch = x[index: index+bs,:]
    y_batch = Y_test[index: index+bs,:]

    z = w_avg.dot(x_batch.T).T
    softmax = np.exp(z)
    softmax = softmax/np.sum(softmax, axis =1).reshape(-1,1)
    predict = np.argmax(softmax, axis =1)
    l = (-1/bs)*np.trace(y_batch.dot(np.log(softmax).T))
    predict_test.extend(predict.tolist())
    cross_entropy_test.append(l)

# 
#Plotting testing cross entropy loss for each classifier
f,ax = plt.subplots()
ax.plot(range(len(cross_entropy_test)), cross_entropy_test)
plt.legend()
plt.title('Testing cross entropy loss (with average weights) vs iterations (batch)')
plt.ylabel('Testing loss')
plt.xlabel('Iterations')
plt.show()
predict_test = np.array(predict_test)

#Testing 3D cloud
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
colors = ['green','black','cyan', 'brown','yellow']
labels = [1004, 1100, 1103, 1200, 1400]

label_conv = {0:1004, 1:1100, 2:1103, 3:1200, 4:1400}
for i in range(len(predict_test)):
    predict_test[i] = label_conv[predict_test[i]]
predict_test = np.array(predict_test)
print('Testing cross entropy loss: {}'.format(len(y_test[predict_test!=y_test])/(len(y_test))))
for index, label in enumerate(labels):

    test_label = test[(predict_test.reshape(-1,) == label) ]
    ax.scatter(test_label[:,0], test_label[:,1], test_label[:,2], c = colors[index])

plt.show()



# ################################################################################################################
# #TRAINING
# print("Training loss calculations.............")
# #Training loss and point cloud
# y_train = data[:,4]
# x_train = data[:,5:]

# #Encoding
# y_encoded_train = np.zeros((len(data), num_class))
# for c in range(num_class):
#     y_encoded_train[:,c] = calc_label(y_train, c)

# #Variables
# train_loss = []
# train_predict = []

# #Calculating predicted values, cross entropy error and loss for training data
# batch = 100
# index =0
# while(index+batch < len(data)):
# 	l = np.zeros(num_class)
# 	hypothesis = np.zeros((batch,num_class))
# 	for c in range(num_class):

# 		l[c] = (1/batch)*0.5*np.sum((avg_weights[c].dot(x_train[index:index+batch,:].T).T - y_encoded_train[index:index+batch,c])**2)
# 		hypothesis[:,c] = avg_weights[c].dot(x_train[index:index+batch,:].T).T
# 	train_predict.extend([label_conv[x] for x in np.argmax(hypothesis, axis = 1)])
# 	train_loss.append(l)

# 	index+=batch

# if(index < len(data)):
# 	batch = len(data)-index
# 	l = np.zeros(num_class)
# 	hypothesis = np.zeros((batch, num_class))
# 	for c in range(num_class):
# 		l[c] = (1/batch)*0.5*np.sum((avg_weights[c].dot(x_train[index:index+batch,:].T).T - y_encoded_train[index:index+batch,c])**2)
# 		hypothesis[:,c] = avg_weights[c].dot(x_train[index:index+batch,:].T).T
# 	train_predict.extend([label_conv[x] for x in np.argmax(hypothesis, axis = 1)])
# 	train_loss.append(l)	


# train_predict = np.array(train_predict)
# cross_entropy_error = (len(train_predict[train_predict != y_train]))/len(train_predict)
# print("Cross entropy training error : {}".format(cross_entropy_error))	
# train_loss = np.array(train_loss)


# #Plotting training loss for each classifier
# label_ = ['classifier1', 'classifier2', 'classifier3', 'classifier4', 'classifier5']
# f,ax = plt.subplots()
# for c in range(0,1):
# 	ax.plot(range(len(train_loss[:,c])), train_loss[:,c], label = label_[c])
# plt.legend()
# plt.title('Training loss (with average weights) vs iterations')
# plt.ylabel('Training loss')
# plt.xlabel('Iterations')
# plt.show()

# #Training 3D cloud
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
# colors = ['green','black','cyan', 'brown','yellow']

# for index, label in enumerate(labels):

#     train_label = data[(train_predict.reshape(-1,) == label) ]
#     ax.scatter(train_label[:,0], train_label[:,1], train_label[:,2], c = colors[index])

# plt.show()


