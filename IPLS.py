# -*- coding: utf-8 -*-
import json 
import socket
import struct 
import time
import random
import numpy as np
import os
from os.path import expanduser
from os import path

class IPLS:
	#initialize the IPLS instance, giving the address of the 
	# IPLS daemon.
	def __init__(self,port = 12000,host = "127.0.0.1"):
		self.port = port;
		self.host = host;


	# In this method, ipls.init is performed by sending a request to the IPLS daemon who
	# 	executes the actual init method. Upon completion, the daemon informs the application
	#	to continue
	def init(self,Path,fileName,bootstraper_list,model_size,model,is_bootstraper = False):
		self.model_size = model_size;
		print("Size : " + str(model_size) + " , " + str(Path) + " , " + self.host + " , " + str(self.port))
		#properly initialize variables	
		self.store(model,fileName)
		self.is_bootstraper = is_bootstraper;
		if(is_bootstraper):
			is_bootstraper = 1;
		else:
			is_bootstraper = 0;

		arr = [1,is_bootstraper,len(bootstraper_list)]
		string = '!hhh'
		for bootstraper in bootstraper_list:
			arr.append(len(bootstraper));
			arr.append(bootstraper);
			string+= 'h' + str(len(bootstraper)) + 's'

		arr.append(len(Path));
		arr.append(Path);
		string += 'h' + str(len(Path)) + 's';

		print(fileName)
		arr.append(len(fileName))
		arr.append(fileName);
		string += 'h' + str(len(fileName)) + 's';

		string+='i'
		arr.append(model_size);
		# Form the packet to be sent by the application
		# 	to the IPLS daemon through IPC
		packet = struct.pack(string,*arr);
		
		# Send the message and wait until completion of the 
		# server
		sockfd = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		sockfd.connect((self.host,self.port));
		sockfd.sendall(packet);
		sockfd.recv(1);
		sockfd.close()


	# In this method, the application sends a request to the 
	# IPLS daemon in order to perform the UpdateModel method.
	# The application waits until completion of the UpdatedModel
	def UpdateModel(self,model):
		packet = struct.pack('!h' + len(model)*'d',2,*model);
		sockfd = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		sockfd.connect((self.host,self.port));
		sockfd.sendall(packet);
		sockfd.recv(1);
		sockfd.close();

		
	# In this method, the application sends a request to the 
	# IPLS daemon in order to perform the GetPartitions method.
	# Upon sending the request, the daemon replies with the updated
	# model
	def GetPartitions(self):
		packet = struct.pack('!h',3);
		sockfd = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		sockfd.connect((self.host,self.port));
		sockfd.sendall(packet);

		updated_model  = sockfd.recv(self.model_size*8)
		while(len(updated_model) != self.model_size*8):
			updated_model  += sockfd.recv(self.model_size*8);
		
		updated_model = struct.unpack('!' + self.model_size*'d',updated_model)
		
		sockfd.close()
		return list(updated_model);


	
	def get_sample(self,X_train,Y_train,batch_size):
		indexes = random.sample(range(0, len(X_train)), batch_size);
		X_sample = []
		Y_sample = []
		for idx in indexes:
			X_sample.append(X_train[idx])
			Y_sample.append(Y_train[idx])

		return X_sample,Y_sample;

	def norm(self,x):
		n = 0;
		for i in range(len(x)):
			n += x[i]
		return n;

	def diff(self,x,y):
		d = []
		for i in range(len(x)):
			d.append(x[i] - y[i])
		return d;

	# Write an initial model into a file 
	def Strore_Initial_Model(self,file,model):
		home = expanduser("~")
		if(path.exists(home+"/IPLS") ==  False):
			os.mkdir(home+"/IPLS")

		data = struct.pack('!' + len(model)*'d',*model)
		fd = open(home+"/IPLS/" + file.decode(),'wb');
		fd.write(data);
		fd.close();

	def store(self,model,file):
		weights = self.get_model_parameters(model);
		self.Strore_Initial_Model(file,weights)


	def fit(self,model,X_train,Y_train,batch_size,iterations):
		if(self.is_bootstraper):
			print("bootstraper must wait until training is finished")
			while(1):
				pass
		for i in range(iterations):
			
			X,Y = self.get_sample(X_train,Y_train,batch_size);
			
			old_weights = self.GetPartitions()
			print("NORM OF MODEL : " + str(self.norm(old_weights)))
		
			#old_weights = self.get_model_parameters(model)
			self.set_model_parameters(old_weights,model);
			
			model.fit(np.array(X),np.array(Y),batch_size,1);
			
			new_weights = self.get_model_parameters(model);
			
			print("UPDATED NORM : " + str(self.norm(new_weights)))
			
			#update = self.diff(old_weights,new_weights);
			print("ok")
			self.UpdateModel(new_weights);
			
				
	#Return the number of parameters and bias weights on 
	# each layer. 

	def find_params(self,shape):
		params = 1;
		for dim in shape:
			params *= dim;
		return params,shape[len(shape)-1]

	#This fucntion takes as input the model and returns 
	# an 1D array of the model weights so that they can be 
	# taken as input to IPLS
	
	def get_model_parameters(self,model):
	    params = [0 for i in range(model.count_params())]
	    counter = 0;
	    for layer in model.layers:
	        weights = layer.get_weights();
	        for i in range(len(weights)):
	            n,bias = self.find_params(np.shape(weights[i]))
	            x = np.reshape(weights[i],(n))
	            for i in range(counter,counter+n):
	                params[i] = x[i-counter];
	            counter += n;
	    if(counter != len(params)):
	        print("Error in serialization")
	    return params

	#This function takes as input a list of parameters
	# and changes with that list the weights of the model
	def set_model_parameters(self,parameters,model):
		new_params = []
		counter = 0;
		layer = 0
		mapping = []
		for i in range(len(model.layers)):
			weights = model.layers[i].get_weights();
			if(len(weights) > 0):
				arr = [];
				for i in range(len(weights)):
					arr.append(weights[i].shape);
				mapping.append(arr);  
			else:
				mapping.append([]);

		for shape in mapping:
			if(len(shape) > 0):
				arr = []
				for lshape in shape:
					params,b = self.find_params(lshape);
					arr.append(np.reshape(parameters[counter:(counter+params)],lshape))
					counter += params;
				model.layers[layer].set_weights(arr)
				layer+=1
			else:
				new_params.append([]);
				layer += 1


