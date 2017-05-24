import theano
import theano.tensor as T
import lasagne

from lasagne.random import get_rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class RepeatLayer(lasagne.layers.Layer):
    def __init__(self, incoming, n, **kwargs):
        super(RepeatLayer, self).__init__(incoming, **kwargs)
        self.n = n

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], None)

    def get_output_for(self, input, **kwargs):
        return  T.tile(input.reshape((input.shape[0], input.shape[1], 1)), (1,1,self.n))

class RandomPad(lasagne.layers.Layer):
    def __init__(self, incoming, n=4, **kwargs):
        super(RandomPad, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.n = n

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1]*self.n, input_shape[2])

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true noise is disabled, see notes
        """
        if deterministic:
            return T.repeat(input,self.n,axis=1)
        else:
            idx = T.cast(T.sort(self._srng.uniform(size=(input.shape[0],input.shape[1]*self.n),low=0,high=input.shape[1]-0.001),axis=1),'int32')
            bi = T.arange(input.shape[0]).reshape((input.shape[0],1))
            
            return input[bi,idx]
