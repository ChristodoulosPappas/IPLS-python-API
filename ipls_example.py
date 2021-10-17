import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import random
import time
import sys
import os
from IPLS import IPLS
from os.path import expanduser
from os import path


#In order to run this example first start IPFS and IPLS daemon.
# The you are free to execute it, for example : 
# python3 ipls_example.py 0 4 /ip4/127.0.0.1/tcp/5001 1 12D3KooWCyJZJphf9z1Dbd2sJKYc11PVV2RBVA9HQjNz26oMANgR
# Where the first input (0) is the dataset chunck that the application is going to use for training also the _id of the peer,
# the second one (4) is the number of datasets, the root dataset is going to be partitioned
# the third one is the IPFS API address where the IPLS daemon is going to communicate with the IPFS daemon
# the forth one indicates if the peer is bootstrapper 1 or not 0.
# the final one is the bootstraper IPFS id. 
# Thus in order to run this example in your pc, first of all start as many peers (lets say n peers) as you wish, by  
# starting n different ipfs daemons. Then start n different IPLS daemons by running Middleware.java, with port  
# numbers 12000 through 12000 + n. Finally run n applications (ipls_example.py) like : 
# python3 ipls_example.py 0 n /ip4/127.0.0.1/tcp/5001 1 your_bootstraper
# python3 ipls_example.py 1 n /ip4/127.0.0.1/tcp/5002 0 your_bootstraper
# ....
# python3 ipls_example.py n-1 n /ip4/127.0.0.1/tcp/(5000 + n) 0 your_bootstraper



_id = int(sys.argv[1]);
number_of_peers = int(sys.argv[2])
IPFS_addr = sys.argv[3].encode()
is_bootstraper = int(sys.argv[4])
bootstraper = sys.argv[5].encode()
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
ipls.init(IPFS_addr,("init_model_" + str(_id)).encode(),[bootstraper],model.count_params(),model,is_bootstraper);

ipls.fit(model,X, Y, batch_size, 1000)

