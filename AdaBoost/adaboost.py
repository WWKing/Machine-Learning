# coding: UTF-8
__author__ = 'Wking'

from numpy import *

#加载数据以及标签
def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#单层决策树的分类器，根据输入的值与阀值进行比较得到输出结果，因为是单层决策树，所以只能比较数据一个dimen的值
#threshVal阈值、threshIneq大于或者小于标记、dimen维度
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

#构造单层决策树，这部分的构造的思路和前面的决策树是一样的，只是这里的评价体系不是熵而是加权的错误率，
# 这里的加权是通过数据的权重D来实现的，每一次build权重都会因上一次分类结果不同而不同。返回的单层决策树的相关信息存在字典结构中方便接下来的使用
#D为权重向量
def buildstump(dataArr,classLabel,D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabel).T
    m,n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {} #用于存储单层决策树的信息
    bestClasEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j)*stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                #加权错误率
                weightedError = D.T * errArr
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if(weightedError < minError):
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    #存储最佳决策树信息
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['inequal'] = inequal
    #返回最佳决策树信息、最小错误率、最佳分类结果
    return bestStump,minError,bestClasEst

#基于单层决策树的AdaBoost训练过程
def adaBoostTrainsDS(dataArr,classLabels,numIt = 40):
    weakClassArr = []
    m = shape(dataArr)[0]
    #初始化所有样本的权值一样
    D = mat(ones((m,1))/m)
    #每个数据点的估计值
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,ClassEst = buildstump(dataArr,classLabels,D)
        #计算alpha，max(error,1e-16)保证没有错误的时候不出现除零溢出
        #alpha表示的是这个分类器的权重，错误率越低分类器权重越高
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #print "ClassEst:",ClassEst.T
        #为下一次迭代计算D
        expon = multiply(-1*alpha*mat(classLabels).T,ClassEst) #exponent for D calc, getting messy
        D = multiply(D,exp(expon))                              #Calc New D for next iteration
        D = D/D.sum()
        #记录每个点的类别估计累计值
        aggClassEst += alpha*ClassEst
        #print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        #最终的错误率
        errorRate = aggErrors.sum()/m
        #print "total error: ",errorRate
        if errorRate == 0.0:
            break
    return weakClassArr


def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['inequal'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        #print aggClassEst
    return sign(aggClassEst)


if __name__ == "__main__":
    dataArr,labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray = adaBoostTrainsDS(dataArr,labelArr,30)
    testArr,testlabelArr = loadDataSet('horseColicTest2.txt')
    result = adaClassify(testArr,classifierArray)
    errArr = mat(ones((67,1)))
    print result
    print errArr[result != mat(testlabelArr).T].sum()