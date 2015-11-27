# coding: UTF-8
__author__ = 'Wking'
from math import log
import  operator

#计算数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelcounts = {}
    for featvVec in dataSet:
        currentLabel = featvVec[-1]
        if currentLabel not in labelcounts.keys():
            labelcounts[currentLabel] = 0
        labelcounts[currentLabel]+=1
    shannoEnt = 0.0
    for key in labelcounts:
        prob = float(labelcounts[key])/numEntries
        shannoEnt -= prob*log(prob,2)
    return  shannoEnt

#按照给定特征划分数据集
def splitDataset(dataSet,axis,value):
    retDataSet =[]
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    baseShannoEntry = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featureList = [example[i] for example in dataSet]
        uniqueVals = set(featureList)  #创建唯一的分类标签
        newEntropy = 0.0
        #对每个特征划分一次数据集
        for val in uniqueVals:
            subDataSet = splitDataset(dataSet,i,val)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseShannoEntry - newEntropy
        if infoGain>bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return  bestFeature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

#创建树代码
def createtree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    #类别完全相同，停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #遍历完所有特征时返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #选择最佳属性
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)
    for val in uniqueValues:
        sublabels = labels[:]
        myTree[bestFeatLabel][val] = createtree(splitDataset(dataSet,bestFeat,val),sublabels)
    return  myTree

def Classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = Classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel



#使用决策树的分类函数
if __name__ == "__main__":
    dateSet,labels = createDataSet();
    myTree = createtree(dateSet,labels)
    print(myTree)

