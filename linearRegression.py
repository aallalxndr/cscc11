import math

def mean(values):
    mean = sum(values) / float(len(values))
    return mean

def covariance(x, meanX, meanY):
    covariance = 0.0
    for i in range(len(x)):
        covariance +=(x[i] - meanX)* (y[i] - meanY)
    return covariance

def variance(values, mean):
    variance = sum([(x-mean)**2 for x in values])
    return variance

def coefficients(data):
    x = [row[0] for row in data]
    y = [row[1] for row in data]
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x , x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return [b0,b1]

def regress(train,test):
    predictions = list()
    b0, b1 = coefficients(train)
    for row in test:
        y = b0 + b1 * row[0]
        predictions.append(y)
    return predictions