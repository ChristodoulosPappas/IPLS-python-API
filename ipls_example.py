import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import random
import time
import sys
import os
from IPLS import IPLS
import os
from os.path import expanduser
from os import path

#In order to run this example first start IPFS and IPLS daemon.
# The you are free to execute it, for example : 
# python3 ipls_example.py 0 4 /ip4/127.0.0.1/tcp/5001 1 12D3KooWCyJZJphf9z1Dbd2sJKYc11PVV2RBVA9HQjNz26oMANgR
# Where the first input (0) is the dataset chunck that the application is going to use for training,
# the second one (4) is the number of datasets, the root dataset is going to be partitioned
# the third one is the IPFS API address where the IPLS daemon is going to communicate with the IPFS daemon
# the forth one is the id of the peer (this is only done for this application so that multiple peers can run in the same machine)
# the final one is the bootstraper IPFS id. 



def get_sample(X_train,Y_train,batch_size):
	indexes = random.sample(range(0, len(X_train)), batch_size);
	X_sample = []
	Y_sample = []
	for idx in indexes:
		X_sample.append(X_train[idx])
		Y_sample.append(Y_train[idx])

	return X_sample,Y_sample;

def norm(x,y):
	n = 0;
	for i in range(len(x)):
		n += (y[i] - x[i])*(y[i] - x[i]);
	return n

def diff(x,y):
	d = []
	for i in range(len(x)):
		d.append(x[i] - y[i])
	return d;

def fit(model,X_train,Y_train,batch_size,iterations):
	for i in range(iterations):
		X,Y = get_sample(X_train,Y_train,batch_size);
		old_weights = get_model_parameters(model)
		model.fit(np.array(X),np.array(Y),batch_size,1);
		new_weights = get_model_parameters(model)
		print(norm(new_weights,old_weights))
		update = diff(old_weights,new_weights);
		time.sleep(1)
        


#Return the number of parameters and bias weights on 
# each layer. 
def find_params(shape):
	params = 1;
	for dim in shape:
		params *= dim;
	return params,shape[len(shape)-1]



#This fucntion takes as input the model and returns 
# an 1D array of the model weights so that they can be 
# taken as input to IPLS
def get_model_parameters(model):
	params = [0 for i in range(model.count_params())]
	counter = 0;
	for layer in model.layers:
		weights = layer.get_weights();
		if(len(weights) != 0):
			n,bias = find_params(np.shape(weights[0]))
			x = np.reshape(weights[0],(n))
			for i in range(counter,counter+n):
				params[i] = x[i-counter];
			counter += n;
			for i in range(counter,counter+bias):
				params[i] = weights[1][i- counter]
			counter += bias
    
	return params

#This function takes as input a list of parameters
# and changes with that list the weights of the model
def set_model_parameters(parameters,model):
	new_params = []
	mapping = []
	counter = 0;
	layer = 0
	for i in range(len(model.layers)):
		weights = model.layers[i].get_weights();
		if(len(weights) > 0):
			mapping.append(weights[0].shape);  

	print(mapping)
	for shape in mapping:
		if(len(shape) > 0):
			params,bias = find_params(shape)
			model.layers[layer].set_weights([np.reshape(parameters[counter:(counter+params)],shape),np.array(parameters[(counter+params):(counter+params+bias)])])
			counter += params + bias
			layer+=1
		else:
			new_params.append([]);
			layer += 1
    



_id = int(sys.argv[1]);
number_of_peers = int(sys.argv[2])
IPFS_addr = bytes(sys.argv[3])
is_bootstraper = int(sys.argv[4])
bootstraper = bytes(sys.argv[5])
if(is_bootstraper == 0):
	is_bootstraper = False;
else:
	is_bootstraper = True;





num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

num_pixels = 784
x_train = x_train.reshape(x_train.shape[0],
                         num_pixels)
x_test = x_test.reshape(x_test.shape[0],
                         num_pixels)



X = []
Y = []
# Segment the dataset so that peers can have different datasets. Note that this code is 
# for experiments
for i in range( int(60000/number_of_peers )*  _id , int(60000/number_of_peers) *  (_id+1)):
	X.append(x_train[i]);
	Y.append(y_train[i])

X = np.array(X)
print(len(X))

print(x_train.shape)
# Scale images to the [0, 1] range
X = X.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
Y = keras.utils.to_categorical(Y, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#create the model
model = keras.Sequential([
  layers.Dense(64, activation='relu', input_shape=(784,)),
  layers.Dense(64, activation='relu'),
  layers.Dense(10, activation='softmax'),
])

model.summary()


batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#Create the IPLS instance
ipls = IPLS(12000 + _id)

#call the init and then the fit (with 1000 iterations) to train the model
ipls.init(IPFS_addr,bytes("init_model_" + str(_id)),[bootstraper],model.count_params(),model,is_bootstraper);

ipls.fit(model,X, Y, batch_size, 1000)

