import math

def squareerror(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += prediction_error**2
    mean = sum_error/float(len(actual))
    return sqrt(mean)

def evaluate(data, algorithm):
    