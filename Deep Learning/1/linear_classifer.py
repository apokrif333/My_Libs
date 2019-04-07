from scipy.optimize import approx_fprime
import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''

    sum_all = np.sum(np.exp(predictions))

    return np.apply_along_axis(lambda x: np.e ** x / sum_all, 0, predictions)


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''

    return -np.log(probs[target_index])


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    predictions = predictions.astype('float64')
    dprediction = np.zeros(predictions.shape)
    loss = 0
    loss_count = 0

    reduced_predictions = predictions - np.max(predictions)
    if reduced_predictions.ndim > 1:
        for index in range(len(reduced_predictions)):
            soft = softmax(reduced_predictions[index])
            loss += cross_entropy_loss(soft, target_index[index])
            loss_count += 1

            f = lambda x: -np.log(np.e ** x[target_index[index]] / np.sum(np.exp(x))) / len(reduced_predictions)
            dprediction[index] = approx_fprime(reduced_predictions[index], f, epsilon=1e-6)

    else:
        soft = softmax(reduced_predictions)
        loss += cross_entropy_loss(soft, target_index)
        loss_count += 1

        f = lambda x: -np.log(np.e ** x[target_index] / np.sum(np.exp(x)))
        dprediction = approx_fprime(reduced_predictions, f, epsilon=1e-6)

    # print(f"I'm loss {loss / loss_count}, I'm grad {dprediction}")
    return loss / loss_count, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    raise Exception("Not implemented!")

    return loss, grad
    

def linear_softmax(X: object, W: object, target_index: object) -> object:
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    print(X, W)

    predictions = np.dot(X, W)
    loss = 0
    loss_count = 0
    dW = np.zeros(W.transpose().shape)

    reduced_predictions = predictions - np.max(predictions)
    if reduced_predictions.ndim > 1:
        for index in range(len(reduced_predictions)):
            soft = softmax(reduced_predictions[index])
            loss += cross_entropy_loss(soft, target_index[index])
            loss_count += 1

            f = lambda x: -np.log(np.e ** x[target_index[index]] / np.sum(np.exp(x))) / len(reduced_predictions)
            dW[index] = approx_fprime(reduced_predictions[index], f, epsilon=1e-6)

    print(loss / loss_count, dW.transpose())
    return loss / loss_count, dW.transpose()


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            raise Exception("Not implemented!")

            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        raise Exception("Not implemented!")

        return y_pred



                
                                                          

            

                
