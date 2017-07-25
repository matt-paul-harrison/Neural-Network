
import numpy as np
import matplotlib.pyplot as plt
import random as rand

## This code implements a generic feed-forward neural network
## It has been written based on code introduced at the link below:
## http://neuralnetworksanddeeplearning.com/chap1.html

cost_function="Cross Entropy"
class Neural_Network:
    def RESET(self):
        ##This function resets the weights and biases.
        self.biases=[np.random.randn(y,1) for y in self.network_dims[1:]]
        self.weights=[np.random.randn(y,x)/self.network_dims[0] for x,y in zip(self.network_dims[:-1],self.network_dims[1:])]
        self.score_list=[]
    
    def __init__(self,eta=1,lmda=.05,MINI_BATCH_SIZE=10**2*5,network_dims=[38,30,2]):
        self.eta=eta                         ###Learning rate eta
        self.lmda=lmda                       ###Regularization parameter eta
        self.MINI_BATCH_SIZE=MINI_BATCH_SIZE ###How many samples from train_data to use at any one time
        self.network_dims=network_dims       ###Should be a list (input dimension), (layer 1 node number), ...., (output dimension)
        self.nlayers=len(network_dims)
        self.cost_function=cost_function
        ##Here we initialize the weights and biases
        self.biases=[np.random.randn(y,1) for y in self.network_dims[1:]]
        self.weights=[np.random.randn(y,x)/self.network_dims[0] for x,y in zip(self.network_dims[:-1],self.network_dims[1:])]
        self.score_list=[]

        
    if cost_function=="Cross Entropy":
        def fn(self,a, y):
            jj=0
            ww=np.array(self.weights)**2
            for i in ww:
                jj+=np.sum(i)
            return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))+self.lmda/2*len(y)*jj
        def Delta(self,z, a, y): 
            return (a-y)

        def cost_derivative(self,output_activations, y):
            #cost function is 1/2(y-C(x))^2
            return (output_activations-y)
    elif cost_function=="Quadratic":
        def fn(self,a, y):
            return np.sum(.5*(y-a)**2)
        def Delta(self,z, a, y): 
            return (a-y)*sigmoid_prime(z)

        def cost_derivative(self,output_activations, y):
            #cost function is 1/2(y-C(x))^2
            return (output_activations-y)
    else:
        print("ERROR: Unrecognized cost function option")

    def evaluate(self,test_data):
        #test_results = [.5*((feedforward(x))- y)**2
        #                for (x, y) in test_data]
        test_results=[]
        for (x,y) in test_data:
            a=self.feedforward(x)
            test_results.append(self.fn(a,y))
        return np.sum(test_results)/len(test_results)
    
    #### Here is the neuron activation function, taken to be the sigmoid function by default
    #### A change in neuron activation function could be achieved by changing these functions
    #### However, be sure to verify that cost_derivative function is changed accordingly
    def sigmoid(self,z):
        #"""The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self,z):
        #"""Derivative of the sigmoid function."""
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def feedforward(self,a):
        #"""Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a)+b)
        return a
    
    ##The two functions below implement the updating of the weights and biases
    ##For details on how this algorithm is derived, see the URL at the top of this class
    def update_wandb(self,input_data,eta,biases,weights):
        nabla_b=[np.zeros((b.shape)) for b in self.biases]
        nabla_w=[np.zeros((w.shape)) for w in self.weights]
        for x, y in input_data:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        weights = [w*(1-self.lmda*self.eta/len(input_data))-(eta/len(input_data))*nw for w, nw in zip(weights, nabla_w)]
        biases = [b-(self.eta/len(input_data))*nb for b, nb in zip(biases, nabla_b)]
        return biases,weights
    def backprop(self,x,y):
        nabla_b=[np.zeros((b.shape)) for b in self.biases]
        nabla_w=[np.zeros((w.shape)) for w in self.weights]
        activation = x
        activations = [x]
        zs=[]
        for b, w in zip(self.biases,self.weights):
            z=np.dot(w,activation)+b
            zs.append(z)
            activation=self.sigmoid(z)
            activations.append(activation)
        delta=self.Delta(zs[-1],activations[-1],y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for ll in np.arange(2,self.nlayers):
            z=zs[-ll]
            sp=self.sigmoid_prime(z)
            delta = np.dot(self.weights[-ll+1].transpose(),delta)*sp
            nabla_b[-ll]=delta
            nabla_w[-ll]=np.dot(delta,activations[-ll-1].transpose())
        return (nabla_b,nabla_w)
    def Dropper(self,N,layer=1):
        ## Takes N Neurons from the desired layer and sets their biases to a huge number.
        ## For a sigmoid activation function this effectively sets their output to zero
        ## This is to be used for dropout.
        ## The biases of the neurons that were knocked out and their addresses are returned as stored_biases and xloc, respectively
        ## If these variables are passed to function unDropper it will undo this operation
        xloc=[1]*N+[0]*(node_number-N)
        rand.shuffle(xloc)
        stored_biases=(np.array(biases[layer-1]).flatten())*np.array(xloc).transpose()
        for i in np.arange(len(xloc)):
            if xloc[i]==1:
                biases[layer-1][i]+=10**8
        return stored_biases,xloc

    def unDropper(self,stored_biases,xl,layer=1):
        xlocin=~np.array(xl)+2
        biases[0]=(np.array(biases[layer-1]).flatten())*np.array(xlocin).transpose()+stored_biases
        biases[0]=biases[layer-1].reshape(len(biases[layer-1]),1)
        
####################################################################################################################################        
## Data fed to this algorithm should be an array of length 2 vectors.  e.g.:
# len(test_data) = (number of testing examples)
# len(test_data[0]) = 2
# len(test_data[0][0]) = (length of input vector) = network_dims[0]
# len(test_data[0][1]) = (lenght of output vector) = network_dims[-1]
cost_function="Cross Entropy"     ##Options for cost function are "Cross Entropy" and "Quadratic"
train_data=np.load("example_training_data.npy") #loads an example of training data
test_data=np.load("example_test_data.npy")  #loads an example of testing data
Network1=Neural_Network(MINI_BATCH_SIZE=10) ## Instantiate Neural_Network class, options below
# eta -> learning rate
# lmda -> regularization rate (adds w**2 to cost function for updating purposes)
# MINI_BATCH_SIZE -> number of training samples to consider at any one time
# network_dims -> vector whose elements are integers: [(input length),(number of nodes in layer 1)...,(output length)]
iterations=10**2  #number of training iterations
epoch_length=10**1 #number of iterations before randomly generating a new mini_batch
for i in range(iterations):
    if i%epoch_length==0:
        ## Every epoch, shuffle test and training data, then select a mini_batch from them
        rand.shuffle(train_data)
        rand.shuffle(test_data)
        train_mini_batch=train_data[:Network1.MINI_BATCH_SIZE]
        test_mini_batch=test_data[:Network1.MINI_BATCH_SIZE]
        if i>0:
            score=Network1.evaluate(test_mini_batch)
            Network1.score_list.append(score)
            print("Score is "+str(score))

    Network1.biases,Network1.weights=Network1.update_wandb(train_mini_batch,Network1.eta,Network1.biases,Network1.weights)
plt.plot(Network1.score_list)
plt.ylabel('score')
plt.xlabel('number of epochs')
#### To generate plots of estimated output vs actual output, use the Network1.feedforward(x) function
#### test_data[i][1] is the true output, while Network1.feedforward(test_data[i][0]) is the network's prediction
