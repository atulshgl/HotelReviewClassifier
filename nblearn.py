import math, sys, os, json, re, string
import collections as clt

#textFilePath = sys.argv[1]
#labelFilePath = sys.argv[2]
outputFilePath = os.path.join(os.path.dirname(__file__)) + "/nbmodel.txt"
labelFilePath = os.path.join(os.path.dirname(__file__)) + "/dataCorpus/train-labels.txt"
textFilePath = os.path.join(os.path.dirname(__file__)) + "/dataCorpus/train-text.txt"

def readLabels (filePath):
    codeLabelMapping = dict()
    with open(filePath, 'r') as doc:
        for line in doc:
            col = line.strip().split(' ')
            codeLabelMapping[col[0]] = [col[1],col[2]]                         
    return codeLabelMapping
    
def readReviews (filePath):
    codeReviewMapping = dict()
    with open(filePath,'r') as doc:
        for line in doc:
            idx = line.find(' ')
            code = line[:idx]
            review = line[idx:].lower().strip(string.punctuation)#.translate(None, string.punctuation)
            codeReviewMapping[code] = review
    return codeReviewMapping

def divideIntoSets(labelMap, reviewMap,size):
    trainingMap = dict()
    devMap = dict()
    i=0
    for code,labels in labelMap.items():
        if i<size: 
            trainingMap[code] = [reviewMap[code],labels[0],labels[1]]
        else:
            devMap[code] = [reviewMap[code],labels[0],labels[1]]
        i = i+1
    return (trainingMap, devMap)

def CalculateParameters(trainingMap):
    pr1 = clt.Counter()
    pr2 = clt.Counter()
    vocab = dict()
    l = dict()
    instanceCount = clt.defaultdict(clt.Counter)
    for code,col in trainingMap.items():
        pr1[col[1]] += 1
        pr2[col[2]] += 1
        l[col[1]],l[col[2]] = 1,1
        for word in re.findall('[A-Za-z]+',col[0]):
            instanceCount[col[1]][word] += 1
            instanceCount[col[2]][word] += 1
            vocab[word] = 1
    deno = 0
    for key in pr1.keys():
        deno += pr1[key]
    prior1 = dict()
    for key in pr1.keys():
        prior1[key] = pr1[key]/float(deno)
    deno = 0
    for key in pr2.keys():
        deno += pr2[key]
    prior2 = dict()
    for key in pr2.keys():
        prior2[key] = pr2[key]/float(deno)

    posts = clt.defaultdict(dict)
    vocabSize = len(vocab)
    for label,p in l.items():
        denom = float(sum(instanceCount[label].values()) + vocabSize)
        for word in vocab:
            numer = 1
            if word in instanceCount[label]:
                numer += instanceCount[label][word]
            posts[label][word] = math.log(numer/denom)
    return (prior1,prior2,posts)

### Read Label and text files and convert the data
### into desired format for faster computation
codeLabelMap = readLabels(labelFilePath) 
codeReviewMap = readReviews(textFilePath)

### Use partRatio % of the reviews for training set ###
### Use remaining reviews for the development set  ###
partRatio = 0.25
reviewSize = min(len(codeLabelMap),len(codeReviewMap))
trainingSetSize = int(reviewSize*partRatio)

(trainingMap, devMap) = divideIntoSets(codeLabelMap,codeReviewMap,trainingSetSize)
    
### Calculate the priors of classes and posteriori
### probabilities for all the words and label combinations
### Also do laplace smooting + logarithm of probabilities
(prior1,prior2,posts) = CalculateParameters(trainingMap)

with open(outputFilePath, 'w') as f:
    json.dump(prior1, f)
    f.write("\n")
    json.dump(prior2, f)
    f.write("\n")
    json.dump(posts, f)

