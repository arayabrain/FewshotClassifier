import lasagne
import theano
import theano.tensor as T
from lasagne.layers import helper

### Permutation Equivariant layer
#
# To include this layer in a model, the input should be a rank 3 tensor of the form (BATCH, FEATURES, OBJECTS)
#
# You must also specify the network contained within the permutation equivariant wrapper. To do this, create
# a separate InputLayer with input shape (BATCH,2*FEATURES,OBJECTS,OBJECTS) and then create a network using
# Lasagne's NINLayer in place of DenseLayer. This way, the dense network is applied identically and in parallel to
# all interaction pairs. The reason that the second dimension is 2*FEATURES is that it concatenates features from
# two objects. The final layer of that network should be provided as the 'subnet' argument.
#
# Finally, pooling is applied on the trailing dimension. This is specified by the 'pooling' argument, which can be
# one of 'mean', 'max', or a custom theano function (which should perform a reduction over the fourth dimension of the 
# input tensor).
###########################################3

class PermutationalLayer(lasagne.layers.Layer):
	def __init__(self,incoming,subnet,pooling='mean',**kwargs):
		super(PermutationalLayer, self).__init__(incoming, **kwargs)
		self.subnet = subnet
		self.pooling = pooling

	def get_output_for(self, input, **kwargs):
		rs = input.reshape((input.shape[0], input.shape[1], input.shape[2], 1)) # B,V,S,1
		z1 = T.tile( rs, (1,1,1,input.shape[2]))
		z2 = z1.transpose((0,1,3,2))
		Z = T.concatenate([z1,z2],axis=1)
		Y = helper.get_output(self.subnet, Z)
		if self.pooling == 'mean':
			return T.mean(Y,axis=3)
		elif self.pooling == 'max':
			return T.max(Y,axis=3)
		else: return self.pooling(Y)

	def get_params(self, **tags):
		# Get all parameters from this layer, the master layer
		params = super(PermutationalLayer, self).get_params(**tags)
		# Combine with all parameters from the child layers
		params += helper.get_all_params(self.subnet, **tags)
		return params

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], self.subnet.output_shape[1], input_shape[2])
