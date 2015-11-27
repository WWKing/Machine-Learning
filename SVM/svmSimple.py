# coding: UTF-8
__author__ = 'Wking'
from numpy import *

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

#随机选择数据，i为alpha的下标，m为alpha的数目
def selectJrand(i,m):
    j = i;
    while(j == i):
        j = int(random.uniform(0,m))
    return j

#调整大于H或者小于L的alpha值
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

#简化版SMO算法
#dataMatIn数据集、classLabels数据标签、C常数、toler容错率、MaxIter最大迭代次数
def smoSimple(dataMatIn,classLabels,C,toler,MaxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))  #初始化alpha
    iter = 0
    while(iter<MaxIter):
        alphaPairsChanged = 0
        for  i in range(m):
            FXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = FXi - float(labelMat[i])
            #如果alpha可以更改，进入优化过程
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                FXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = FXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                #保证alpha在o到c之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print "L==H"; continue
                #eta为alpha[j]的最优修改量
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print "eta>=0"; continue
                #修改alpha[j]
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                #判断alpha[j]是否有轻微改变
                if (abs(alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; continue
                #update i by the same amount as j ,the update is in the oppostie direction
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                #设置常数项b
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged+=1
                print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print "iteration number: %d" % iter
    return b,alphas

if __name__ == "__main__":
    dataArr,labels = loadDataSet("testSet.txt")
    b,alphas = smoSimple(dataArr,labels,0.6,0.001,40)
    print b
    #输出支持向量
    for i in range(100):
        if alphas[i] > 0.0:
            print dataArr[i],labels[i]



