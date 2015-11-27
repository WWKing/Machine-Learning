# coding: UTF-8
__author__ = 'Wking'
from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#Logistic回归梯度上升优化算法
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    m,n = shape(dataMatrix)
    alpha = 0.001   #步长
    maxCycles = 500 #最大迭代次数
    weights = ones((n,1))
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        #按照差值的方向调整回归系数
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights

#Logistics回归随机梯度上升优化算法
def StocGradAscent0(dataMatrix,classLabels):
    m,n =shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i]-h
        weights = weights + alpha *  error * dataMatrix[i]
    return weights

#Logistics回归随机梯度上升优化算法改进
def StocGradAscent1(dataMatrix,classLabels,numIter = 150):
    m,n =shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0 + i +j) +0.01
            randomIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randomIndex]*weights))
            error = classLabels[randomIndex]-h
            weights = weights + alpha *  error * dataMatrix[randomIndex]
            del(dataIndex[randomIndex])
    return weights



#画出数据集和logistic回归最佳拟合直线
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()


if __name__ == "__main__":
    dataMatIn,classLabels = loadDataSet()
    weights1 = StocGradAscent0(array(dataMatIn),classLabels)
    weights2 = StocGradAscent1(array(dataMatIn),classLabels,150)
    weights3 = gradAscent(dataMatIn,classLabels)
    #weights.getA()返回自己，但是以ndarray方式返回
    plotBestFit(weights1)
    plotBestFit(weights2)
    plotBestFit(weights3.getA())
