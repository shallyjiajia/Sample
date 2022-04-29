import hanlp

hanlp.pretrained.tok.ALL
hanlp.pretrained.pos.ALL
hanlp.pretrained.ner.ALL
hanlp.pretrained.dep.ALL
hanlp.pretrained.mtl.ALL

def test():
    HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
    doc = HanLP(['2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。', '阿婆主来到北京立方庭参观自然语义科技公司。'])
    print(doc)
    doc.pretty_print()
    print(doc.to_conll())


def test2():
    HanLP = hanlp.pipeline() \
        .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
        .append(hanlp.load('FINE_ELECTRA_SMALL_ZH'), output_key='tok') \
        .append(hanlp.load('CTB9_POS_ELECTRA_SMALL'), output_key='pos') \
        .append(hanlp.load('MSRA_NER_ELECTRA_SMALL_ZH'), output_key='ner', input_key='tok') \
        .append(hanlp.load('CTB9_DEP_ELECTRA_SMALL', conll=0), output_key='dep', input_key='tok') \
        .append(hanlp.load('CTB9_CON_ELECTRA_SMALL'), output_key='con', input_key='tok')
    doc = HanLP('2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。')
    print(doc)
    doc.pretty_print()
    print(doc.to_conll())


def test3():
    def gettok(sents, whitelist):
        tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        tok.dict_force = whitelist

        data = tok(sents)

        return data

    sents = ['维修人员检查检查门边框，发现门边框胶条脱落。']
    whitelist = ['门边框', '门边框胶条']
    data = gettok(sents, whitelist)
    print(data)

if __name__ == '__main__':
    test3()
