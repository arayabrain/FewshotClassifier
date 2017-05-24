import numpy as np

import lasagne
from lasagne.layers import Gate
from lasagne.init import Orthogonal
import theano
import theano.tensor as T

from math import *
import pickle

import time

from channel_layers import RepeatLayer, RandomPad

""" Task and network parameters """

# Task dimension
DIM = 64
MINSEP = 1.0
MAXSEP = 2.0

# Task sparseness
SPARSE = 6

def getRandomParams():
	vec1 = np.zeros(DIM)
	idx = np.random.permutation(DIM)
	#if SPARSE:
	for i in range(SPARSE):
		vec1[idx[i]] = np.random.randn()

	vec1 = vec1/np.sqrt(np.sum(vec1**2)+1e-32)

	vec2 = np.random.randn()

	vec2 = vec2/np.sqrt(np.sum(vec2**2)+1e-32)

	sig1 = np.random.randn(DIM,DIM)/sqrt(DIM)
	sig2 = np.random.randn(DIM,DIM)/sqrt(DIM)
	mean = MINSEP+(MAXSEP-MINSEP)*np.random.rand()
	return vec1,vec2,sig1,sig2,mean

def mkData(N, Nt):
    vec1,vec2,sig1,sig2,mean = getRandomParams()
    
    data = []
    tdata = []
    tlabels = []
    
    for i in range(N):
		s = np.random.randn(DIM)
		lbl = np.random.randint(2)
		example = np.matmul(s,(lbl==0)*sig1+(lbl==1)*sig2) + mean*(2*lbl-1)*vec1 + vec2
		example = np.hstack([example,np.array([2*lbl-1])])
		data.append(example)

    for i in range(Nt):
		s = np.random.randn(DIM)
		lbl = np.random.randint(2)
		example = np.matmul(s,(lbl==0)*sig1+(lbl==1)*sig2) + mean*(2*lbl-1)*vec1 + vec2
		example = np.hstack([example,np.array([0])])
		tdata.append(example)
		tlabels.append(lbl)
    
    data = np.array(data)
    tdata = np.array(tdata)
    tlabels = np.array(tlabels)
    
    return data,tdata,tlabels

invar = T.tensor3()
invar2 = T.tensor3()
targ = T.matrix()

train_input = lasagne.layers.InputLayer((None,None,DIM+1), input_var = invar)

inp_shape = invar.shape[1]
test_shape = invar2.shape[1]

test_input = lasagne.layers.InputLayer((None,None,DIM+1), input_var = invar2)

train_shuf = lasagne.layers.DimshuffleLayer(train_input,(0,2,1))

test_shuf = lasagne.layers.DimshuffleLayer(test_input,(0,2,1))

def mkBlock(inp):
    nin1 = lasagne.layers.NINLayer(inp, num_units = 64)
    nin2 = lasagne.layers.NINLayer(nin1, num_units = 64)
    pool = lasagne.layers.GlobalPoolLayer(nin2)
    res = lasagne.layers.ReshapeLayer(pool, (-1,64,1))
    rep = RepeatLayer(res, n=inp_shape)
    return lasagne.layers.ConcatLayer([inp,rep])

cur = train_shuf
for i in range(3):
    cur = mkBlock(cur)

pfinal = lasagne.layers.NINLayer(cur, num_units = 256)
gpool = lasagne.layers.GlobalPoolLayer(pfinal)

teacher_reshape = lasagne.layers.ReshapeLayer(gpool, (-1,256,1))
teacher_repeat = RepeatLayer(teacher_reshape, n=test_shape)
teacher_stack = lasagne.layers.ConcatLayer([test_shuf, teacher_repeat], axis=1)

t_classify1 = lasagne.layers.NINLayer(teacher_stack, num_units = 256)
t_classify2 = lasagne.layers.NINLayer(t_classify1, num_units = 256)
t_classify3 = lasagne.layers.NINLayer(t_classify2, num_units = 256)
t_stack = lasagne.layers.ConcatLayer([t_classify1, t_classify3], axis=1)
t_classify4 = lasagne.layers.NINLayer(t_stack, num_units = 256)
t_final = lasagne.layers.NINLayer(t_classify4, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
t_output = lasagne.layers.DimshuffleLayer(t_final,(0,2,1))

t_out = lasagne.layers.get_output(t_output)
t_params = lasagne.layers.get_all_params(t_output, trainable=True)

t_elemloss = T.mean((-targ*T.log(t_out[:,:,0]*0.99998+1e-5) - (1-targ)*T.log(1-t_out[:,:,0]*0.99998-1e-5) ),axis=1)
t_loss = T.mean(t_elemloss)

grad = T.sum(theano.grad(t_loss, invar)**2,axis=(1))
grad2 = T.sum(theano.grad(t_loss, invar2)**2,axis=(1))
gloss = T.mean( (grad-1)**2, axis=(0,1) ) + T.mean( (grad2-1)**2, axis=(0,1) )

t_updates = lasagne.updates.adam(t_loss + gloss, t_params, learning_rate = 1e-3)

t_train = theano.function([invar, invar2, targ], t_loss, updates=t_updates, allow_input_downcast=True) 

mns = MINSEP
mxs = MAXSEP
sv = SPARSE

for epoch in range(2000):
	MINSEP = mns+2*(1-epoch/2000.0)
	MAXSEP = mxs+2*(1-epoch/2000.0)
	SPARSE = int(DIM-(DIM-sv)*epoch/2000)
		
	err = 0
	NT = 40
	N = 40
	data = []
	tdata = []
	tlabels = []
	for i in range(800):
		d,td,l = mkData(N,NT)
		data.append(d)
		tdata.append(td)
		tlabels.append(l)
	data = np.array(data)
	tdata = np.array(tdata)
	tlabels = np.array(tlabels)

	e0 = t_train(data,tdata,tlabels)

	f = open("pretrain.txt","a")
	f.write("%d %.6g\n" % (epoch, e0))
	f.close()

for epoch in range(2000,4000):
	MINSEP = mns
	MAXSEP = mxs
	SPARSE = sv
		
	err = 0
	NT = 40
	N = 40
	data = []
	tdata = []
	tlabels = []
	for i in range(800):
		d,td,l = mkData(N,NT)
		data.append(d)
		tdata.append(td)
		tlabels.append(l)
	data = np.array(data)
	tdata = np.array(tdata)
	tlabels = np.array(tlabels)

	e0 = t_train(data,tdata,tlabels)

	f = open("pretrain.txt","a")
	f.write("%d %.6g\n" % (epoch, e0))
	f.close()

pickle.dump(lasagne.layers.get_all_param_values(gpool), open("teacher.params","wb"))
