import pandas as pd
import csv
from nlp.nlpprocessor import NLPProcessor

class NLPBean():
    def __init__(self):
        self.appcontext = None

        self.nlp = NLPProcessor()
        self.path1 = ""
        self.path2 = ""

    def to_df2(self, docs):
        dfs = []
        for i in range(len(docs)):
            sent = pd.DataFrame()
            doc = docs[i]
            for k in ['tok/fine', 'pos/ctb', 'ner/msra', 'dep']:
                if k == 'ner/msra':
                    if len(doc[k]) > 0:
                        for j in range(len(doc['tok/fine'])):
                            for v in doc[k]:
                                if v[0] == doc['tok/fine'][j]:
                                    sent.loc[j, k] = v[1]
                                    break
                    else:
                        sent[k] = '-'
                elif k == 'dep':
                    lid = []
                    ldeprel = []
                    for v in doc[k]:
                        lid.append(v[0])
                        ldeprel.append(v[1])
                    sent[k + '/id'] = lid
                    sent[k + '/deprel'] = ldeprel
                else:
                    sent[k] = doc[k]
            sent = sent.fillna('-')
            dfs.append(sent)

        return dfs

    def getphrase(self, dfs):

        def getp(df, cindex, pstr, pid):

            if cindex in pid[:-1]:
                return ''

            if df.loc[cindex, 'ner/msra'] in ['EQP', 'A', 'C']:
                pstr = pstr + df.loc[cindex, 'tok/fine']

            if df.loc[cindex, 'dep/id'] == 0:
                return pstr

            elif df.loc[cindex, 'dep/id'] == len(df):
                return pstr

            elif cindex == df.loc[cindex, 'dep/id']:
                return pstr
            else:
                cindex = df.loc[cindex, 'dep/id']
                pid.append(cindex)
                return getp(df, cindex, pstr, pid)

        plist = []
        for i in range(len(dfs)):
            df = dfs[i]
            try:
                df1 = df[df['ner/msra'] == 'EQP']
                p = []
                for i in df1.index:
                    phrase = getp(df, i, '', [])
                    p.append(phrase)
                plist.append(p)
            except Exception as e:
                print(i)
                print(e)

        return plist

    def savephrase(self, plist):
        with open('defectphrase.csv', 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(plist)
            f.close()

    def getstructdata(self, dfs):
        slist = []
        for i in range(len(dfs)):
            df = dfs[i]
            try:
                eqp, a, c = set(), set(), set()
                for i in df.index:
                    label = df.loc[i, 'ner/msra']
                    value = df.loc[i, 'tok/fine']
                    if label == 'EQP':
                        eqp.add(value)
                    elif label == 'A':
                        a.add(value)
                    elif label == 'C':
                        c.add(value)
                    else:
                        continue

                s = pd.concat([pd.DataFrame({'EQP': list(eqp)}),
                               pd.DataFrame({'A': list(a)}),
                               pd.DataFrame({'C': list(c)})])
                s = s.fillna('-')
                slist.append(s)
            except Exception as e:
                print(i)
                print(e)

        return slist

    def getDoc(self):
        docs = []
        count = 0
        for d in self.nlp.data:
            doc = {}
            try:
                doc['tok/fine'] = d
                doc['pos/ctb'] = self.nlp.posTagging(d)
                doc['ner/msra'] = self.nlp.nameEntityRecognize(d)
                doc['dep'] = self.nlp.dparse(d)
                docs.append(doc)
            except Exception as e:
                count += 1

        return docs

    #path1 : data file, '../data/更新后词文件.csv'
    #path2 : tagging file, '../data/更新后标注文件.csv'
    def loadDataFromFile(self):
        dt1 = pd.read_csv(self.path1, header=None)
        dt2 = pd.read_csv(self.path2, header=None)
        data = []
        whitelist = {}
        vocab = {}

        n = len(dt1)

        for i in range(n):
            l1 = dt1.iloc[i, 0].split(' ')
            l2 = dt2.iloc[i, 0].split(' ')
            data.append(l1)

            for j in range(len(l1)):
                key = l1[j]
                if (key not in whitelist) and (l2[j] in ['EQP', 'A', 'C']):
                    whitelist[key] = l2[j]
                    vocab[key] = l2[j]

        self.nlp.loadData(data)
        self.nlp.setVocab(vocab)
        self.nlp.setWhitelist(whitelist)

    def initNLPStrategy(self):
        self.nlp.loadStrategy()

    def initialize(self):
        self.loadDataFromFile()
        self.initNLPStrategy()

if __name__ == '__main__':
    nlptest = NLPBean()

    nlptest.path1='../data/更新后词文件.csv'
    nlptest.path2='../data/更新后标注文件.csv'
    #nlptest.loadDataFromFile()
    #nlptest.initNLPStrategy()
    nlptest.initialize()

    docs = nlptest.getDoc()
    print(docs)
    dfs = nlptest.to_df2(docs)
    plist = nlptest.getphrase(dfs)
    slist = nlptest.getstructdata(dfs)
    nlptest.savephrase(plist)




