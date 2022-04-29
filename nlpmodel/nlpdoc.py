# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 10:09:12 2022

@author: shally jia
"""

import hanlp

hanlp.pretrained.tok.ALL
hanlp.pretrained.pos.ALL
hanlp.pretrained.ner.ALL
hanlp.pretrained.dep.ALL
hanlp.pretrained.mtl.ALL
import pandas as pd
import csv


class NLPDoc():
    def __init__(self):
        """
        tok:
            input ['商品和服务。', '阿婆主来到北京立方庭参观自然语义科技公司']
            ouput [['商品', '和', '服务', '。'], ['阿婆主', '来到', '北京立方庭', '参观', '自然语义科技公司']]
        ner:
            input  [["2021年", "HanLPv2.1", ""带来", "NLP", "技术", "。"], ['阿婆主', '来到', '北京立方庭', '参观', '自然语义科技公司']]
            output [[('2021年', 'DATE', 0, 1)], [('北京', 'LOCATION', 2, 3), ('立方庭', 'LOCATION', 3, 4), ('自然语义科技公司', 'ORGANIZATION', 5, 9)]]
        pos:
            input ["我", "的", "希望", "是", "希望", "张晚霞", "的", "背影", "被", "晚霞", "映红", "。"]
            output ['PN', 'DEG', 'NN', 'VC', 'VV', 'NR', 'DEG', 'NN', 'LB', 'NR', 'VV', 'PU']
        dep:
            input ["2021年", "HanLPv2.1", "为", "生产", "环境", "带来", "次", "世代", "最", "先进", "的", "多", "语种", "NLP", "技术", "。"]
            output [(6,'tmod'),(6,'nsubj'),(6,'prep'),(5,'nn'),(3,'pobj'),(0,'root'),(8,'det'),(15,'nn'),(10,'advmod'),(15,'rcmod'),(10,'cpm'),(13,'nummod'),(15,'nn'),(15,'nn'),(6,'dobj'),(6,'punct')]

        """
        self.data = []
        self.whitelist = {}
        self.tags = {}
        self.loaddata()
        self.tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        self.ner = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)
        self.dep = hanlp.load(hanlp.pretrained.dep.CTB9_DEP_ELECTRA_SMALL, conll=0)
        self.pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)

    def loaddata(self):

        path1 = '../data/更新后词文件.csv'
        path2 = '../data/更新后标注文件.csv'
        dt1 = pd.read_csv(path1, header=None)
        dt2 = pd.read_csv(path2, header=None)

        #        dt1 = dt1[:20]
        #        dt2 = dt2[:20]

        n = len(dt1)

        for i in range(n):
            l1 = dt1.iloc[i, 0].split(' ')
            l2 = dt2.iloc[i, 0].split(' ')
            self.data.append(l1)
            for j in range(len(l1)):
                key = l1[j]
                if (key not in self.whitelist) and (l2[j] in ['EQP', 'A', 'C']):
                    self.whitelist[key] = l2[j]

    def to_df(doc, ner='msra', tok='fine', pos='ctb'):
        ktok = 'tok/' + tok
        kpos = 'pos/' + pos
        kner = 'ner/' + ner
        keys = [ktok, kpos, kner, 'dep']

        df = []
        for i in range(len(doc[ktok])):
            sent = pd.DataFrame()
            for k in keys:
                if k == kner:
                    for v in doc[k][i]:
                        for j in range(len(doc[ktok][i])):
                            if v[0] == doc[ktok][i][j]:
                                sent.loc[j, k] = v[1]
                elif k == 'dep':
                    lid = []
                    ldeprel = []
                    for v in doc[k][i]:
                        lid.append(v[0])
                        ldeprel.append(v[1])
                    sent[k + '/id'] = lid
                    sent[k + '/deprel'] = ldeprel
                else:
                    sent[k] = doc[k][i]
                sent = sent.fillna('-')
            df.append(sent)

        return df

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

    def settok(self):
        self.tok.dict_force = self.whitelist

    def setner(self):
        self.ner.dict_whitelist = self.whitelist
        self.ner.dict_tags = self.tags

    def getdoc(self):
        """
        1，分词
        2，词性标注
        3，命名实体标注
        4，语义依存分析
        """
        self.setner()

        docs = []
        count = 0
        for d in self.data:
            doc = {}
            try:
                doc['tok/fine'] = d
                doc['pos/ctb'] = self.pos(d)
                doc['ner/msra'] = self.ner(d)
                doc['dep'] = self.dep(d)
                docs.append(doc)
            except Exception as e:
                #                print(len(d))
                #                print(self.data.index(d))
                #                print(e)
                count += 1
        print(count)

        return docs

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
        with open('../result/故障短语.csv', 'w', encoding='utf-8') as f:
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





if __name__ == '__main__':
    nlpdoc = NLPDoc()
    #     nlpdoc.ner = ner()
    docs = nlpdoc.getdoc()
    dfs = nlpdoc.to_df2(docs)
    plist = nlpdoc.getphrase(dfs)
    slist = nlpdoc.getstructdata(dfs)
    #     x = nlpdoc.whitelist
    nlpdoc.savephrase(plist)
#     print(df)




