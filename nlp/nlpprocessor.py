import pandas as pd
import csv


class NLPProcessor():
    def __init__(self):
        self.data = []
        self.whitelist = {}
        self.tags = {}
        self.vocab = {}

        self.tokenizer = None
        self.ner = None
        self.postagger = None
        self.dep = None

    #Params sentences:['阿婆主来到北京立方庭参观自然语义科技公司','...']
    #Return result:[['阿婆主', '来到', '北京立方庭', '参观', '自然语义科技公司'],[...]]
    def tokenize(self,sentences):
        return self.tokenizer(sentences)

    #Params words:[["我", "的", "希望", "是", "希望", "张晚霞", "的", "背影", "被", "晚霞", "映红", "。"],[...]]
    #Return result:[['PN', 'DEG', 'NN', 'VC', 'VV', 'NR', 'DEG', 'NN', 'LB', 'NR', 'VV', 'PU'],[...]]
    def posTagging(self,words):
        return self.postagger(words)

    #Params [["2021年", "HanLPv2.1", ""带来", "NLP", "技术", "。"], ['阿婆主', '来到', '北京立方庭', '参观', '自然语义科技公司']]
    #Return [[('2021年', 'DATE', 0, 1)], [('北京', 'LOCATION', 2, 3), ('立方庭', 'LOCATION', 3, 4),
    def nameEntityRecognize(self,words):
        return self.ner(words)

    #Params words:["2021年", "HanLPv2.1", "为", "生产", "环境", "带来", "次", "世代", "最", "先进", "的", "多", "语种", "NLP", "技术", "。"]
    #Return [(6, 'tmod'), (6, 'nsubj'), (6, 'prep'), (5, 'nn'), (3, 'pobj'), (0, 'root'), (8, 'det'), (15, 'nn'), (
    #10, 'advmod'), (15, 'rcmod'), (10, 'cpm'), (13, 'nummod'), (15, 'nn'), (15, 'nn'), (6, 'dobj'), (6, 'punct')]
    def dparse(self,words):
        return self.dep(words)

    def setWhitelist(self,whitelist):
        self.whitelist = whitelist

    def setVocab(self,vocab):
        self.vocab = vocab

    def setTags(self,tags):
        self.tags = tags

    def loadData(self,data):
        self.data = data

    def loadStrategy(self):
        #load packages
        #for HanLP
        import hanlp
        hanlp.pretrained.tok.ALL
        hanlp.pretrained.pos.ALL
        hanlp.pretrained.ner.ALL
        hanlp.pretrained.dep.ALL
        hanlp.pretrained.mtl.ALL
        #for Baidu
        #from LAC import LAC
        #from ddparser import DDParser

        #Dependency Parser
        self.dep = None
        #for HanLP
        self.dep = hanlp.load(hanlp.pretrained.dep.CTB9_DEP_ELECTRA_SMALL, conll=0)
        #for Baidu
        #ddp = DDParser(use_pos=True)
        #self.dep = ddp.parse_seg

        #NER
        self.ner = None
        #for HanLP
        self.ner = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)
        self.ner.dict_whitelist = self.vocab
        self.ner.dict_tags = self.tags

        #POSTAG
        self.postagger = None
        #for HanLP
        self.postagger = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
        # for Baidu
        # lac = LAC(mode='lac')
        # self.postagger = lac.run

        #TOK
        self.tokenizer = None
        #for HanLP
        self.tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        self.tokenizer.dict_force = self.vocab
        #for Baidu
        #lac = LAC(mode='seg')
        #self.tokenizer = lac.run

if __name__ == '__main__':
    nlp = NLPTest()

    #path1 : data file, '../data/更新后词文件.csv'
    #path2 : tagging file, '../data/更新后标注文件.csv'
    nlp.loadDataFromFile('../data/更新后词文件.csv','../data/更新后标注文件.csv')

    nlp.loadStrategy()

    ################################################################

    docs = nlp.composeDoc()
    print(docs)

    dfs = nlp.to_df2(docs)
    plist = nlp.getphrase(dfs)
    slist = nlp.getstructdata(dfs)
    nlp.savephrase(plist)

'''
    #vocab = {'门边框胶条':'EQP','脱落':'A'}
    #nlp.setVocab(vocab)
    #tags = {}
    #nlp.setTags(tags)

    #Tokenize
    sents = ['维修人员检查门边框，发现门边框胶条脱落。','我方人员检查门边框，发现边框胶条脱落。']
    words = nlp.tokenize(sents)
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
'''




