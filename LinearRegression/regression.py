#coding:utf-8
__author__ = 'Wking'

from numpy import *
import matplotlib.pyplot as plt

#加载数据
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))-1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#线性回归
def standRegres(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular,cannot do inverse"
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

#局部加权线性回归
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    #创建对角矩阵
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular,cannot do inverse"
        return
    ws = xTx.I *(xMat.T *(weights * yMat))
    return testPoint * ws

#局部线性回归函数
def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

#岭回归
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print "This matrix is singular,cannot do inverse"
        return
    ws = denom.I*(xMat.T*yMat)
    return ws

def rigeTest(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    #数据标准化
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMeans = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wmat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wmat[i,:] = ws.T
    return wmat

#测试线性回归函数
def testLR():
    xArr,yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr,yArr)
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat*ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy*ws
    ax.plot(xCopy[:,1],yHat)
    plt.show()

#测试局部加权线性回归
def testLELR():
    xArr,yArr = loadDataSet('ex0.txt')
    yHat = lwlrTest(xArr,xArr,yArr,1)
    xMat = mat(xArr)
    yMat = mat(yArr)
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T.flatten().A[0],s=2,c='red')
    plt.show()

#测试岭回归函数
def testRL():
     xArr,yArr = loadDataSet('abalone.txt')
     rigeWights = rigeTest(xArr,yArr)
     fig =  plt.figure()
     ax = fig.add_subplot(111)
     ax.plot(rigeWights)
     plt.show()

if __name__ == "__main__":
    #testLR()
    #testLELR()
    testRL()