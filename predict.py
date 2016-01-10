# import from system
import pdb
import cPickle
# import from third library
import theano
# import from local folder
import data_loader

def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    classifier = cPickle.load(open('best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test test
    dataset='data/mnist.pkl.gz'
    training_set, validation_set, testing_set, = data_loader.load(dataset)
    testing_set_x   , testing_set_y    = testing_set
    testing_set_x = testing_set_x.get_value()
    testing_set_y = testing_set_y.eval()[:30]

    predicted_values = predict_model(testing_set_x[:30])
    print ("Predicted values for the first 10 examples in test set:")
    print predicted_values
    print ("answers:")
    print 

if __name__ == '__main__':
    predict()
