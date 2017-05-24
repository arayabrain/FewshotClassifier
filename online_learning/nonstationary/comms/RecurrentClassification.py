import numpy as np

import lasagne
from lasagne.init import Orthogonal
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from math import *
import pickle

import time

import os
import sys

DIM = 32
BS = 4

SPARSE = 8
MINSEP = 0
MAXSEP = 4

class RepeatLayer(lasagne.layers.Layer):
    def __init__(self, incoming, n, **kwargs):
        super(RepeatLayer, self).__init__(incoming, **kwargs)
        self.n = n

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], None)

    def get_output_for(self, input, **kwargs):
        return  T.tile(input.reshape((input.shape[0], input.shape[1], input.shape[2], 1)), (1,1,1,self.n))

def getRandomParams():
    vec1 = np.zeros(DIM)
    idx = np.random.permutation(DIM)
    for i in range(SPARSE):
        vec1[idx[i]] = np.random.randn()

    vec1 = vec1/np.sqrt(np.sum(vec1**2)+1e-32)

    vec2 = np.random.randn()

    vec2 = vec2/np.sqrt(np.sum(vec2**2)+1e-32)

    sig1 = np.random.randn(DIM,DIM)/sqrt(DIM)
    sig2 = np.random.randn(DIM,DIM)/sqrt(DIM)
    mean = MINSEP+(MAXSEP-MINSEP)*np.random.rand()
    return vec1,vec2,sig1,sig2,mean

def mkData(N): # This is now making N batches of size BS
    vec1,vec2,sig1,sig2,mean = getRandomParams()
    
    data = []
    tdata = []
    tlabels = []
    
    for i in range(2*N*BS):
		s = np.random.randn(DIM)
		lbl = np.random.randint(2)
		example = np.matmul(s,(lbl==0)*sig1+(lbl==1)*sig2) + mean*(2*lbl-1)*vec1 + vec2
		example = np.hstack([example,0])
		data.append(example)
		tlabels.append(lbl)
		if i%2*BS == 0:
			vec1 = vec1 + np.random.randn(DIM)*0.12 - 0.03*vec1
			vec1 = vec1/np.sqrt(np.sum(vec1**2)+1e-32)
			vec2 = vec1 + np.random.randn(DIM)*0.12 - 0.03*vec2
			vec2 = vec2/np.sqrt(np.sum(vec2**2)+1e-32)
			sig1 = sig1 + 0.2*np.random.randn(DIM,DIM)/sqrt(DIM) - 0.02*sig1
			sig2 = sig2 + 0.2*np.random.randn(DIM,DIM)/sqrt(DIM) - 0.02*sig2
    
    data = np.array(data)
    tlabels = np.array(tlabels)
    
    train_batches = []
    test_batches = []
    test_labels = []
    
    for i in range(N):
        batch = []
        for j in range(BS):
            data[2*(i*BS+j),DIM] = tlabels[2*(i*BS+j)]
            batch.append(data[2*(i*BS+j)])
        train_batches.append(batch)
    train_batches = np.array(train_batches).transpose((0,2,1))
    
    for i in range(N):
        batch = []
        lbatch = []
        for j in range(BS):
            batch.append(data[2*(i*BS+j)+1])
            lbatch.append(tlabels[2*(i*BS+j)+1])
            
        test_batches.append(batch)
        test_labels.append(lbatch)
        
    test_batches = np.array(test_batches).transpose((0,2,1))
    test_labels = np.array(test_labels)
    return train_batches, test_batches, test_labels

invar = T.tensor4()
invar2 = T.tensor4()
seqlen = invar.shape[1]

tr_input = lasagne.layers.InputLayer((None,None,DIM+1,BS), input_var = invar)
ts_input = lasagne.layers.InputLayer((None,None,DIM+1,BS), input_var = invar2)

train_input = lasagne.layers.InputLayer((None,DIM+1,BS))

# This block computes the population mean of some function of the input and concatenates that
# to the individual values.
def mkBlock(inp):
    nin1 = lasagne.layers.NINLayer(inp, num_units = 64)
    nin2 = lasagne.layers.NINLayer(nin1, num_units = 64)
    pool = lasagne.layers.GlobalPoolLayer(nin2)
    res = lasagne.layers.ExpressionLayer(pool, function = lambda x: x.reshape((x.shape[0],64,1)).repeat(BS,2), output_shape='auto')
    return lasagne.layers.ConcatLayer([inp,res])

cur = train_input
for i in range(3):
    cur = mkBlock(cur)

# This final pooling layer gives the learned embedding of the task inferred from the dataset
nin_final = lasagne.layers.NINLayer(cur, num_units = 256)
embedding = lasagne.layers.GlobalPoolLayer(nin_final)

# Hidden-to-hidden
emb_input = lasagne.layers.InputLayer((None,256))
l_htoh1 = lasagne.layers.DenseLayer(emb_input, num_units = 64)
l_htoh2 = lasagne.layers.DenseLayer(l_htoh1, num_units = 64)
l_htoh = lasagne.layers.NonlinearityLayer(lasagne.layers.ElemwiseSumLayer([emb_input,lasagne.layers.DenseLayer(l_htoh2, num_units = 256,nonlinearity = None)]), nonlinearity = None)

observer_network = lasagne.layers.CustomRecurrentLayer(tr_input, embedding, l_htoh)

obs_transpose = lasagne.layers.DimshuffleLayer(observer_network, (0,2,1))
sender_msg = lasagne.layers.NINLayer(obs_transpose, num_units = 64, nonlinearity = lasagne.nonlinearities.elu)
sender = lasagne.layers.DimshuffleLayer(sender_msg, (0,2,1))
receiver = lasagne.layers.LSTMLayer(lasagne.layers.GaussianNoiseLayer(sender,sigma=0.1), num_units = 256)

emb_reshape = lasagne.layers.ReshapeLayer(receiver, (-1,seqlen,256,1))
emb_repeat = RepeatLayer(emb_reshape,n=BS)
test_stack = lasagne.layers.ConcatLayer([ts_input, emb_repeat],axis=2)
test_shuf = lasagne.layers.DimshuffleLayer(test_stack,(0,2,3,1))
# Now process each example forwards
classify1 = lasagne.layers.NINLayer(test_shuf, num_units = 256)
classify2 = lasagne.layers.NINLayer(classify1, num_units = 256)
classify3 = lasagne.layers.NINLayer(classify2, num_units = 256)
# Use skip connections for better gradient propagation
classify4 = lasagne.layers.NINLayer(lasagne.layers.ConcatLayer([classify1,classify3]), num_units = 256)

final = lasagne.layers.NINLayer(classify4, num_units=1,nonlinearity=lasagne.nonlinearities.sigmoid)
output = lasagne.layers.DimshuffleLayer(final, (0,3,1,2))

out, messages = lasagne.layers.get_output([output,sender])
params = lasagne.layers.get_all_params(output,trainable=True)

targ = T.tensor3()

regularize = theano.shared(np.cast['float32'](1e-2))
# To avoid NaNs when the network receives an easy problem
out = T.clip(out[:,:,0,:],1e-5,1-1e-5)

elemloss = (-targ*T.log(out) - (1-targ)*T.log(1-out))
loss = T.mean(elemloss)
reg = T.mean(abs(messages))*regularize
updates = lasagne.updates.adam(loss+reg, params, learning_rate = 5e-4)

train = theano.function([invar, invar2, targ], loss, updates=updates, allow_input_downcast=True)
process = theano.function([invar, invar2], [out,messages], allow_input_downcast = True)
test = theano.function([invar, invar2, targ], elemloss, allow_input_downcast=True)

# Uncomment to resume
#lasagne.layers.set_all_param_values(output,pickle.load(open("recurrent_learner.params","rb")))
regularize.set_value(np.cast['float32'](0.05))

def testModel(epoch):
	N=240
	data = []
	tdata = []
	tlabels = []
	for i in range(1600):
		d,td,l = mkData(N)
		data.append(d)
		tdata.append(td)
		tlabels.append(l)
	data = np.array(data)
	tdata = np.array(tdata)
	tlabels = np.array(tlabels)

	outs,msg = process(data,tdata)
	losses = test(data,tdata,tlabels)

	plt.plot(np.mean(losses,axis=(0,2)))
	plt.ylabel("Log-loss")
	plt.xlabel("Batch #")
	plt.savefig("frames/gen%.6d.png" % epoch)
	plt.clf()
	
	plt.imshow(msg[0],vmin=-1,vmax=1,aspect=1, interpolation='nearest', cmap=plt.cm.BuGn)
	plt.gcf().set_size_inches((8,8))
	plt.savefig("frames/msg%.6d.png" % epoch)
	plt.clf()

# Train the classifier network
for epoch in range(10000):
    N = 120
    data = []
    tdata = []
    tlabels = []
    for i in range(400):
        d,td,l = mkData(N)
        data.append(d)
        tdata.append(td)
        tlabels.append(l)
    data = np.array(data)
    tdata = np.array(tdata)
    tlabels = np.array(tlabels)
    
    e0 = train(data,tdata,tlabels)
    
    f = open("error.txt","a")
    f.write("%d %.6g\n" % (epoch, e0))
    f.close()
    
    if epoch%100 == 0:
		testModel(epoch)
		pickle.dump(lasagne.layers.get_all_param_values(output), open("recurrent_learner.params","wb"))
