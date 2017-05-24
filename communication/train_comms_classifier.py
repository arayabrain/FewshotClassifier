import numpy as np

import lasagne
from lasagne.layers import Gate
from lasagne.init import Orthogonal
import theano
import theano.tensor as T

from math import *
import pickle
from permutationlayer import PermutationalLayer

import time

from channel_layers import RepeatLayer, RandomPad
import os
import sys

""" Task and network parameters """

# 0 = Dense, 1 = ConvNet, 2 = LSTM
SPEAK_MODEL = int(sys.argv[1])
LISTEN_MODEL = int(sys.argv[2])

dirnames = ["dense","conv","lstm","msconv"]

subdir = dirnames[SPEAK_MODEL] + "-" + dirnames[LISTEN_MODEL] + "/"

try:
	os.mkdir(subdir)
	os.mkdir(subdir+"pmessages/")
	os.mkdir(subdir+"messages/")
	os.mkdir(subdir+"checkpoints/")
except:
	pass

# Task dimension
DIM = 64
MINSEP = 1.0
MAXSEP = 2.0

# Task sparseness
SPARSE = 6

# Message length
LATENT = 128

# Message noise
CHANNEL_NOISE = 0.5

# Message random padding
PAD_SIZE = 4

# Start with easier problems
PRETRAIN = True

# Continue from checkpoint; either False or the epoch to restart from
CONTINUE = False

def getRandomParams():
	vec1 = np.zeros(DIM)
	idx = np.random.permutation(DIM)
	#if SPARSE:
	for i in range(SPARSE):
		vec1[idx[i]] = np.random.randn()
	#else:
	#	vec1 = np.random.randn()

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
pool2 = lasagne.layers.GlobalPoolLayer(pfinal)

if SPEAK_MODEL == 0:
	msg_dense = lasagne.layers.DenseLayer(pool2, num_units = LATENT, nonlinearity = lasagne.nonlinearities.tanh)
	msg_layer = lasagne.layers.ReshapeLayer(msg_dense,(-1,LATENT,1))
elif SPEAK_MODEL == 1:
	msg_reshape = lasagne.layers.ReshapeLayer(pool2, (-1,256//(LATENT//8),LATENT//8))
	
	msg_conv1 = lasagne.layers.Conv1DLayer(msg_reshape, num_filters = 64, filter_size = 5, pad="same")
	msg_up1 = lasagne.layers.Upscale1DLayer(msg_conv1, scale_factor = 2)
	msg_conv2 = lasagne.layers.Conv1DLayer(msg_up1, num_filters = 64, filter_size = 5, pad="same")
	msg_up2 = lasagne.layers.Upscale1DLayer(msg_conv2, scale_factor = 2)
	msg_conv3 = lasagne.layers.Conv1DLayer(msg_up2, num_filters = 64, filter_size = 5, pad="same")
	msg_up3 = lasagne.layers.Upscale1DLayer(msg_conv3, scale_factor = 2)
	msg_conv4 = lasagne.layers.Conv1DLayer(msg_up3, num_filters = 1, filter_size = 5, pad="same", nonlinearity = lasagne.nonlinearities.tanh)
	msg_layer = lasagne.layers.ReshapeLayer(msg_conv4,(-1,LATENT,1))
elif SPEAK_MODEL == 2:	
	msg_reshape = lasagne.layers.ReshapeLayer(pool2, (-1,256,1))
	msg_repeat = lasagne.layers.DimshuffleLayer(RepeatLayer(msg_reshape, n=LATENT),(0,2,1))
	
	msg_lstm1 = lasagne.layers.LSTMLayer(msg_repeat, num_units = 32)
	msg_lstm2 = lasagne.layers.LSTMLayer(msg_lstm1, num_units = 1, nonlinearity = lasagne.nonlinearities.tanh)
	msg_layer = lasagne.layers.ReshapeLayer(msg_lstm2,(-1,LATENT,1))
	
msg_noise = lasagne.layers.GaussianNoiseLayer(msg_layer,sigma=CHANNEL_NOISE)
msg_pad = RandomPad(msg_noise, n=PAD_SIZE)
#msg_pad = lasagne.layers.ExpressionLayer(msg_noise, function = lambda x: T.tile(x,(1,4,1)), output_shape = 'auto')
msg_shuf = lasagne.layers.DimshuffleLayer(msg_pad,(0,2,1))

if LISTEN_MODEL == 0:
	listen1 = lasagne.layers.DenseLayer(msg_shuf, num_units = 256)
	listen2 = lasagne.layers.DenseLayer(listen1, num_units = 256)
	listen_layer = lasagne.layers.DenseLayer(listen2, num_units = 256)
elif LISTEN_MODEL == 1:
	listen_conv1 = lasagne.layers.Conv1DLayer(msg_shuf, num_filters = 64, filter_size = 9, pad="same")
	listen_pool1 = lasagne.layers.MaxPool1DLayer(listen_conv1, pool_size = 4)
	listen_conv2 = lasagne.layers.Conv1DLayer(listen_pool1, num_filters = 64, filter_size = 9, pad="same")
	listen_pool2 = lasagne.layers.MaxPool1DLayer(listen_conv2, pool_size = 4)
	listen_conv3 = lasagne.layers.Conv1DLayer(listen_pool2, num_filters = 64, filter_size = 9, pad="same")
	listen_pool3 = lasagne.layers.MaxPool1DLayer(listen_conv3, pool_size = 4)
	listen_conv4 = lasagne.layers.Conv1DLayer(listen_pool3, num_filters = 256, filter_size = 9, pad="same")
	listen_layer = lasagne.layers.GlobalPoolLayer(listen_conv4)
elif LISTEN_MODEL == 2:
	listen_layer = lasagne.layers.LSTMLayer(msg_shuf, num_units = 256, only_return_final = True)
elif LISTEN_MODEL == 3: # Multiscale-convolution
	listen_conv1 = lasagne.layers.Conv1DLayer(msg_shuf, num_filters = 32, filter_size = 9, pad="same")
	listen_ds1 = lasagne.layers.Upscale1DLayer(lasagne.layers.MaxPool1DLayer(listen_conv1, pool_size = 2), scale_factor=2)
	listen_ds2 = lasagne.layers.Upscale1DLayer(lasagne.layers.MaxPool1DLayer(listen_conv1, pool_size = 4), scale_factor=4)
	listen_ds3 = lasagne.layers.Upscale1DLayer(lasagne.layers.MaxPool1DLayer(listen_conv1, pool_size = 8), scale_factor=8)
	listen_ds4 = lasagne.layers.Upscale1DLayer(lasagne.layers.MaxPool1DLayer(listen_conv1, pool_size = 16), scale_factor=16)
	listen_stack = lasagne.layers.ConcatLayer([listen_conv1, listen_ds1, listen_ds2, listen_ds3, listen_ds4], axis=1)	
	listen_conv2 = lasagne.layers.Conv1DLayer(listen_stack, num_filters = 32, filter_size = 9, pad="same")
	listen_pool2 = lasagne.layers.MaxPool1DLayer(listen_conv2, pool_size = 4)
	listen_conv3 = lasagne.layers.Conv1DLayer(listen_pool2, num_filters = 64, filter_size = 9, pad="same")
	listen_pool3 = lasagne.layers.MaxPool1DLayer(listen_conv3, pool_size = 4)
	listen_conv4 = lasagne.layers.Conv1DLayer(listen_pool3, num_filters = 256, filter_size = 9, pad="same")
	listen_layer = lasagne.layers.GlobalPoolLayer(listen_conv4)
	
listen_reshape = lasagne.layers.ReshapeLayer(listen_layer, (-1,256,1))
listen_repeat = RepeatLayer(listen_reshape,n=test_shape)

test_stack = lasagne.layers.ConcatLayer([test_shuf, listen_repeat],axis=1)

classify1 = lasagne.layers.NINLayer(test_stack, num_units = 256)
classify2 = lasagne.layers.NINLayer(classify1, num_units = 256)
classify3 = lasagne.layers.NINLayer(classify2, num_units = 256)
stack = lasagne.layers.ConcatLayer([classify1, classify3], axis=1)
classify4 = lasagne.layers.NINLayer(stack, num_units = 256)
final = lasagne.layers.NINLayer(classify4, num_units=1,nonlinearity=lasagne.nonlinearities.sigmoid)
output = lasagne.layers.DimshuffleLayer(final,(0,2,1))

out, message = lasagne.layers.get_output([output,msg_layer])
params = lasagne.layers.get_all_params(output,trainable=True)

elemloss = T.mean((-targ*T.log(out[:,:,0]*0.99998+1e-5) - (1-targ)*T.log(1-out[:,:,0]*0.99998-1e-5) ),axis=1)
loss = T.mean(elemloss)

#if SPEAK_MODEL != 2 and LISTEN_MODEL != 2:
#	grad = T.sum(theano.grad(loss, invar)**2,axis=(2))
#	grad2 = T.sum(theano.grad(loss, invar2)**2,axis=(2))
#	gloss = T.mean( (grad-1)**2, axis=(0,1) ) + T.mean( (grad2-1)**2, axis=(0,1) )
#
#	updates = lasagne.updates.adam(loss+gloss, params, learning_rate = 1e-3)
#else:
updates = lasagne.updates.adam(loss, params, learning_rate = 1e-3)
	
train = theano.function([invar, invar2, targ], loss, updates=updates, allow_input_downcast=True) #,mode=NanGuardMode())
process = theano.function([invar, invar2], out, allow_input_downcast = True)
test = theano.function([invar, invar2, targ], loss,  allow_input_downcast=True)

get_message = theano.function([invar], message, allow_input_downcast=True)

lasagne.layers.set_all_param_values(pool2, pickle.load(open("teacher.params","rb")))

start_epoch = 0

if CONTINUE:
	start_epoch = CONTINUE
	lasagne.layers.set_all_param_values(output, pickle.load(open(subdir+"checkpoints/%.6d.params" % start_epoch,"rb")))
	
from sklearn.manifold import TSNE

def generateExamples(fname):
	msg = np.zeros((0,LATENT,1))

	for j in range(4):
		data = []
		for i in range(400):
			d,td,l = mkData(40,40)
			data.append(d)
		data = np.array(data)

		msg = np.vstack([msg,get_message(data)])

	msg = msg[:,:,0]

	tsn = TSNE(n_components=1)
	tmsg = tsn.fit_transform(msg)

	idx = np.argsort(tmsg[:,0])
	smsg = msg[idx]
	
	np.savetxt(fname,smsg)

if PRETRAIN:
	mns = MINSEP
	mxs = MAXSEP
	sp = SPARSE
	
	SPARSE = 12
	MINSEP = 2
	MAXSEP = 4
	
	for epoch in range(500):
		err = 0
		NT = 40
		N = 40
		data = []
		tdata = []
		tlabels = []
		for i in range(1600):
			d,td,l = mkData(N,NT)
			data.append(d)
			tdata.append(td)
			tlabels.append(l)
		data = np.array(data)
		tdata = np.array(tdata)
		tlabels = np.array(tlabels)

		e0 = train(data,tdata,tlabels)

		f = open(subdir+"pretrain.txt","a")
		f.write("%d %.6g\n" % (epoch, e0))
		f.close()
		if epoch%20 == 0: # Output language examples every 20
			generateExamples(subdir+"pmessages/%.6d.txt" % epoch)

	SPARSE = sp
	MINSEP = mns
	MAXSEP = mxs
	
for epoch in range(start_epoch,start_epoch+10001):
	err = 0
	NT = 40
	N = 40
	data = []
	tdata = []
	tlabels = []
	for i in range(1600):
		d,td,l = mkData(N,NT)
		data.append(d)
		tdata.append(td)
		tlabels.append(l)
	data = np.array(data)
	tdata = np.array(tdata)
	tlabels = np.array(tlabels)

	e0 = train(data,tdata,tlabels)

	f = open(subdir+"error.txt","a")
	f.write("%d %.6g\n" % (epoch, e0))
	f.close()

	if epoch%20 == 0: # Output language examples every 20
		generateExamples(subdir+"messages/%.6d.txt" % epoch)
		
	if epoch%100 == 0: # Output checkpoint every 100
		pickle.dump(lasagne.layers.get_all_param_values(output), open(subdir+"checkpoints/%.6d.params" % epoch,"wb"))
