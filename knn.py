import math
import operator

def euclidenDistance(first,second,length):
    distance = 0
    for x in range(length):
        distance += pow((first[x] - second[x]),2)
    return math.sqrt(distance)

def getNeighbour(train,test,k):
    distances = []
    length = len(train) - 1
    for x in range (len(train)):
        distance = euclidenDistance(test,train[x],length)
        distances.append(train[x],distance)
    distances.sort(key=operator.itemgetter(1))
    neighbours = []
    for x in range (k):
        neighbours.append(distances[x][0])
    return neighbours

def getResponse(neighbours):
    votes = {}
    for x in range(len(neighbours)):
        response = neighbours[x][-1]
        if response in votes:
            votes[response] += 1
        else:
            votes[response] = 1
    sorted = sorted(votes.iteritems(),key=operator.itemgetter(1), reverse = True)
    return sorted[0][0]

def correctness(test,predictions):
    correct = 0
    for x in range(len(test)):
        if test[x][-1] is predictions[x]:
            correct += 1
    return (correct/float(len(test))) * 100.0