import sys, os, math, json, re, string
import collections as clt

#testFilePath = sys.argv[1]
testFilePath = os.path.join(os.path.dirname(__file__)) + "/dataCorpus/test-text.txt"
outputFilePath = os.path.join(os.path.dirname(__file__)) + "/nboutput.txt"
modelFilePath = os.path.join(os.path.dirname(__file__)) + "/nbmodel.txt"
mytestCheckPath = os.path.join(os.path.dirname(__file__)) + "/dataCorpus/train-labels.txt"

def readLabels (filePath):
    codeLabelMapping = dict()
    with open(filePath, 'r') as doc:
        for line in doc:
            col = line.strip().split(' ')
            codeLabelMapping[col[0]] = [col[1],col[2]]                         
    return codeLabelMapping

def readModel(filePath):
    with open(filePath,"r") as model:
        line = model.readlines()
        prior1 = json.loads(line[0])
        prior2 = json.loads(line[1])
        posts = json.loads(line[2])
    return (prior1, prior2, posts)

def readReviews(filePath):
    codeReviewMapping = dict()
    with open(filePath,'r') as doc:
        for line in doc:
            idx = line.find(' ')
            code = line[:idx]
            review = line[idx:].lower().strip(string.punctuation)#.translate(None, string.punctuation)
            codeReviewMapping[code] = review
    return codeReviewMapping

def getStopWords(reviewMap,k):
    wordCount = clt.Counter()
    for code, review in reviewMap.items():
        for word in re.findall('[A-Za-z]+',review):
            wordCount[word] += 1

    stopWords = [word for word,count in wordCount.most_common(k)]
    return stopWords

def classify(review, prior1, prior2, posts,stopWords):
    prob1 = dict()
    for clas in prior1.keys():
        prob1[clas] = math.log(prior1[clas])
        for word in re.findall('[a-z0-9]+',review):
            if word in stopWords:
                continue
            if word not in posts[clas]:
                prob1[clas] = prob1[clas] + math.log(0.001)
            else:
            	prob1[clas] = prob1[clas] + posts[clas][word]
    class1 = max(prob1, key=prob1.get)
    prob2 = dict()
    for clas in prior2.keys():
        prob2[clas] = math.log(prior2[clas])
        for word in re.findall('[a-z0-9]+',review):
            if word in stopWords:
                continue
            if word not in posts[clas]:
                prob2[clas] = prob2[clas] + math.log(0.00001)
            else:
            	prob2[clas] = prob2[clas] + posts[clas][word]
    class2 = max(prob2, key=prob2.get)
    return (class1,class2)

### Read the model from file ###
(prior1, prior2, posts) = readModel(modelFilePath)

### Read the test file ###
testMap = readReviews(testFilePath)
stopWords = getStopWords(testMap,14)
codeLabelMap = readLabels(mytestCheckPath) 

print stopWords

cc = dict()
cc['truthful'], cc['deceptive'] = 0,0
cc['positive'], cc['negative'] = 0,0

dd = dict()
dd['truthful'], dd['deceptive'] = 0,0
dd['positive'], dd['negative'] = 0,0

outMap = dict()
for key, value in testMap.items():
    (class1, class2) = classify(value,prior1,prior2,posts,stopWords)
    outMap[key] = [class1,class2]
    if class1 == codeLabelMap[key][0]:
        cc[class1] += 1
    if class2 == codeLabelMap[key][1]:
        cc[class2] += 1
    dd[codeLabelMap[key][0]] += 1
    dd[codeLabelMap[key][1]] += 1

f1 = cc['truthful']/float(dd['truthful'])
f2 = cc['deceptive']/float(dd['deceptive'])
f3 = cc['positive']/float(dd['positive'])
f4 = cc['negative']/float(dd['negative'])

print cc['truthful'],cc['deceptive'],cc['positive'],cc['negative']
print dd['truthful'],dd['deceptive'],dd['positive'],dd['negative']
print f1, f2, f3, f4
print (f1+f2+f3+f4)/4.0

with open(outputFilePath,'w') as f:
    for k,v in outMap.items():
        f.write(k+ " " + v[0] + " " + v[1] +'\n') 
