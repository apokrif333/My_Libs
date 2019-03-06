from collections import Counter
import numpy as np
import numpy_indexed as npi
import operator


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(X, dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                dists[i_test, i_train] = np.sum(np.abs(X[i_test] - self.train_X[i_train]))

        return dists

    def compute_distances_one_loop(self, X):
        '''
        Computes distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            dists[i_test] = np.apply_along_axis(np.sum, 1, np.abs(X[i_test] - self.train_X))

        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes distance from every sample of X to every training sample
        Fully vectorizes the calculations

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        # Using float32 to to save memory - the default is float64
        dists = np.apply_along_axis(np.sum, 2, np.abs(X[:, None] - self.train_X))

        return dists

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            nearest_k = sorted(dists[i])[:self.k]
            nearest_index = np.where(np.isin(dists[i], nearest_k))[0]
            dict_classes = dict(Counter(self.train_y[nearest_index]))

            if len(dict_classes.keys()) == 2 and dict_classes[False] == dict_classes[True] and \
                    len(nearest_index) == len(nearest_k):
                range_and_classes = np.array([nearest_k, self.train_y[nearest_index]]).astype('int')
                sum_classes = npi.group_by(range_and_classes[1]).sum(range_and_classes[0])

                if sum_classes[1][0] < sum_classes[1][1]:
                    pred[i] = sum_classes[0][0]
                else:
                    pred[i] = sum_classes[0][1]
            else:
                pred[i] = max(dict_classes.items(), key=operator.itemgetter(1))[0]

        return pred

    def predict_labels_multiclass(self, X, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        train_sum = np.array([np.sum(self.train_X, axis=1), self.train_y]).astype('int')
        sum_train_classes = npi.group_by(train_sum[1]).median(train_sum[0])
        test_sum = np.sum(X, axis=1).astype('int')

        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            nearest_k = sorted(dists[i])[:self.k]
            nearest_index = np.where(np.isin(dists[i], nearest_k))[0]
            dict_classes = dict(Counter(self.train_y[nearest_index]))

            same_classes = list(dict_classes.values()).count(max(dict_classes.values()))
            problem_classes = [key for key, value in dict_classes.items() if value == max(dict_classes.values())]

            # if same_classes > 1:
            #
            #     distance = []
            #     for class_y in problem_classes:
            #         distance.append(abs(test_sum[i] - sum_train_classes[1][class_y]))
            #
            #     pred[i] = problem_classes[distance.index(min(distance))]
            #
            # else:
            pred[i] = max(dict_classes.items(), key=operator.itemgetter(1))[0]

        return pred
