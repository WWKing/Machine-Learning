# coding: UTF-8
__author__ = 'Wking'
from numpy import *

#加载数据
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

#完整版platt SMO的支持函数
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #误差缓存first column is valid flag


#计算误差
def calcEK(os,K):
    fXk = float(multiply(os.alphas,os.labelMat).T*(os.X*os.X[K,:].T))+ os.b
    Ek = fXk - float(os.labelMat[K])
    return Ek

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

#选择内循环中的alpha
def selectJ(i,os,Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    #首先将输入值Ei在缓存中设置成为有效的
    os.eCache[i] =[1,Ei]
    #nonzeros(a)返回数组a中值不为零的元素的下标，它的返回值是一个长度为a.ndim(数组a的轴数)的元组，
    ## 元组的每个元素都是一个整数数组，其值为非零元素的下标在对应轴上的值
    #构建一个非零列表
    validEcacheList = nonzero(os.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEK(os, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, os.m)
        Ej = calcEK(os, j)
    return j, Ej

def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEK(oS, k)
    oS.eCache[k] = [1,Ek]

#内层循环
def innerL(i, oS):
    Ei = calcEK(oS, i)
    #如果alpha可以更改，进入优化过程
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print "L==H"; return 0
        #eta为alpha[j]的最优修改量
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0: print "eta>=0"; return 0
        #修改alpha[j]
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        #判断是否为轻微修改
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; return 0
        #update i by the same amount as j  #the update is in the oppostie direction
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i) #added this for the Ecache
        #设置常数项b
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

#platt SMO算法优化版
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    #退出循环的条件：迭代次数大于最大迭代次数并且遍历整个数据集都未对任何alpha值进行修改
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        #一开始遍历任意可能的alpha
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        #遍历所有非边界alpha值
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        if entireSet:
            entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        print "iteration number: %d" % iter
    return oS.b,oS.alphas

#最终的w向量
def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

if __name__ == "__main__":
     dataArr,labels = loadDataSet("testSet.txt")
     b,alphas = smoP(dataArr,labels,0.6,0.001,40)
     ws = calcWs(alphas,dataArr,labels)
     dataMat = mat(dataArr)
     error = 0
     m = shape(dataArr)[0]
     for i in range(shape(dataArr)[0]):
         if dataMat[i]*mat(ws)+b < 0:
             if labels[i] > 0:
                 error += 1
             print "predictlable: %d   realLalel:%d" % (-1,labels[i])
         else:
             if labels[i] < 0:
                 error += 1
             print "predictlable: %d   realLalel:%d" % (1,labels[i])
     print "the total error is: %d the error rate is:%d" %(error,error/m)

