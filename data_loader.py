# import from system
import gzip
import cPickle
# import from third library
import numpy
import theano
import theano.tensor as T

def make_shared(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(value=numpy.asarray(
                                     data_x,
                                     dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(value=numpy.asarray(
                                    data_y,
                                    dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')

def load(dataset):
    f = gzip.open(dataset, 'rb')
    training_set, validation_set, testing_set = cPickle.load(f)
    f.close()
    return make_shared(testing_set), make_shared(validation_set), make_shared(training_set)
