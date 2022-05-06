import pandas as pd
import csv
from nlp.nlpprocessor import NLPProcessor

if __name__ == '__main__':
    nlp = NLPProcessor()

    data = ['维修人员检查门边框，发现门边框胶条脱落。','我方人员检查门边框，发现边框胶条脱落。']

    nlp.loadData(data)

    vocab = {'门边框胶条':'EQP','脱落':'A'}
    nlp.setVocab(vocab)
    tags = {}
    nlp.setTags(tags)

    nlp.loadStrategy()


    #Tokenize
    #sents = ['维修人员检查门边框，发现门边框胶条脱落。','我方人员检查门边框，发现边框胶条脱落。']
    words = nlp.tokenize(data)
    print(words)

    #POS TAG
    postags = nlp.posTagging(words)
    print(postags)

    #NER
    nertags = nlp.nameEntityRecognize(words)
    print(nertags)

    #Dependency Parse
    result = nlp.dparse(words)
    print(result)
