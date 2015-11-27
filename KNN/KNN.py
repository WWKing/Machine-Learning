# coding: UTF-8
__author__ = 'Wking'
from numpy import *
import  operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0][0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#KNN算法的核心代码
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1))-dataSet;
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis = 1)
    distance = sqDistance**0.5
    sortedDistanceIndices = distance.argsort()
    classCount = {}
    for i in range(k):
        votelabel = labels[sortedDistanceIndices[i]]
        classCount[votelabel] = classCount.get(votelabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#将文本数据转换为我们需要的数据类型
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        linesFormatLine = line.split('\t')
        returnMat[index,:] = linesFormatLine[0:3]
        classLabelVector.append(int(linesFormatLine[-1]))
        index+=1
    return returnMat,classLabelVector

#归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet-tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return  normDataSet,ranges,minVals

#针对约会网站测试代码
if __name__ == "__main__":
#def datdingClassTest():
    hoRatio = 0.10 #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount