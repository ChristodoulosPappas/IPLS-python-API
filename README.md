# IPLS-python-API

Because the most famous ML libraries are python libraries, an IPLS API in python was developed, in which the data scientist, instead of using the fit 
from his model instance to train the model, calls the fit method from the python IPLS API. The IPLS fit method takes as input the mode instance, the dataset,
the batch size and the number of iterations. Inside the fit, for each iteration the peer first calls the GetPartitions method, which in turn sends a request the
IPLS daemon to call the actual get_sample method, and return the up to date model parameters. Then the peer calls the model fit method to train the model and
upon completion calls the UpdateModel, which sends a request to the IPLS daemon in order to call the actual IPLS UpdateModel. Note that before
calling ipls.fit, the programmer must initialize the instance, providing the socket port number (default 12000), and the host address (127.0.0.1), 
and then call the init method which requests to the IPLS daemon to call the ipls init method. This repo provides an example (ipls_example.py) of how to use the API.
