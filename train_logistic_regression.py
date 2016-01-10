# import from system
import os
import sys
import pdb
import timeit
import cPickle
# import from third library
import numpy
import theano
from theano import tensor
# import from local folder
import data_loader
from logistic import LogisticRegression
def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='data/mnist.pkl.gz',
                           batch_size=600):

    training_set, validation_set, testing_set, = data_loader.load(dataset)
    training_set_x  , training_set_y   = training_set
    validation_set_x, validation_set_y = validation_set
    testing_set_x   , testing_set_y    = testing_set

    # compute number of minibatches for training, validation and testing
    n_train_batches = training_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = validation_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches  = testing_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = tensor.lscalar()

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = tensor.matrix('x')
    y = tensor.ivector('y')

    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: testing_set_x[index * batch_size: (index + 1) * batch_size],
            y: testing_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: validation_set_x[index * batch_size: (index + 1) * batch_size],
            y: validation_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = tensor.grad(cost=cost, wrt=classifier.W)
    g_b = tensor.grad(cost=cost, wrt=classifier.b)

    # update the parameters of the model
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: training_set_x[index * batch_size: (index + 1) * batch_size],
            y: training_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is # found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = 5 * n_train_batches # requency of training

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            # iter: number of minibatches used)
            iter = epoch * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    # update best_validation_loss
                    best_validation_loss = this_validation_loss
                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('best_model.pkl', 'w') as f:
                        cPickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break
        epoch = epoch + 1

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))

if __name__ == '__main__':
    sgd_optimization_mnist()
