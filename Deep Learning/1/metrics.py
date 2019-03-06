import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, accuracy, f1 - classification metrics
    '''
    accuracy = 0

    # TODO: implement metrics!
    only_true = np.count_nonzero(np.isclose(prediction, ground_truth) == 1)
    false_pos = np.count_nonzero(prediction.astype('int') - ground_truth.astype('int') == 1)
    false_neg = np.count_nonzero(prediction.astype('int') - ground_truth.astype('int') == -1)

    accuracy = only_true / len(ground_truth)
    precision = only_true / (only_true + false_pos)
    recall = only_true / (only_true + false_neg)
    f1 = 2 * (precision * recall) / (precision + recall)
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    only_true = np.count_nonzero(np.isclose(prediction, ground_truth) == 1)

    return only_true / len(ground_truth)
