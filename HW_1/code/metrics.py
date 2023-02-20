import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            if y_pred[i] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if y_pred[i] == 1:
                FP += 1
            else:
                FN += 1

    print(TP, TN, FP, FN)

    precision = 0
    if (TP + FP) != 0:
        precision = TP / (TP + FP)
    print("Precision =", precision)

    recall = 0
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    print("Recall =", recall)

    f1 = 0
    if precision != 0 and recall != 0:
        den = 1/precision + 1/recall
        if den != 0:
            f1 = 2/den
    print("F1 =", f1)

    accuracy = 0
    if (TP + TN + FP + FN) != 0:
        accuracy = (TP + TN) / (TP + TN + FP + FN)
    print("Accuracy =", accuracy)

    return precision, recall, f1, accuracy



def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    correct = 0
    all = len(y_pred)
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            correct += 1
    return correct/all


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    return 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    return np.sum((y_true - y_pred)**2)/len(y_true)


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    return np.sum(np.absolute(y_true - y_pred))/len(y_true)
