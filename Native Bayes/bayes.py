# coding: UTF-8
__author__ = 'Wking'
from numpy import *

#加载数据
def loadDataSet():
     postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
     classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
     return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

#将文档转换为向量
def setOfWord2Vec(vocablist,inputSet):
    returnVec = [0]*len(vocablist)
    for word in inputSet:
        if(word in inputSet):
            returnVec[vocablist.index(word)] = 1
        else: print("the word s% is not in my Vocabulary!" % word)
    return  returnVec

#朴素贝叶斯分类器训练函数
#计算得到文档属于类别为1的概率，以及文档属于1（0）类别的情况下各个单词出现的概率
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs) #文档类别为1的出现概率。注意sum(trainCategory)计算得到的是类别为1的文档总数
    p0Num = ones(numWords) #类别为0的文档中相应词汇的出现次数
    p1Num = ones(numWords) #类别为1的文档中相应词汇的出现次数
    p0Demo = 2.0 #类别为0的文档的单词总数
    p1Demo = 2.0 #类别为1的文档的单词总数
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Demo += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Demo += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Demo) #类别为0的文档中各单词出现的概率
    p0Vect = log(p0Num/p0Demo) #类别为1的文档中各单词出现的概率
    return p0Vect,p1Vect,pAbusive

#vec2Classify:需要进行分类的文本
#p0Vec：类别为0的文档中各单词出现的概率
#p1Vec：类别为1的文档中各单词出现的概率
#pClass1：类别为1的文档出现概率
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1-pClass1)
    if(p1 > p0):
        return 1
    else:
        return 0

#改进词集模型为词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

if __name__ == "__main__":
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)









