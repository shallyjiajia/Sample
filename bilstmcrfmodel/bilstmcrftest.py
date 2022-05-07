import pickle
import re

import numpy as np
import torch

# 加载模型
model = torch.load('model/model0.pkl')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 加载数据
with open('data/datasavetest.pkl', 'rb') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)

# 记载自定义字典
f = open("data/dict_self.txt", encoding='utf-8')
lines = f.readlines()
dict = []
sort_index = []
for line in lines:
    dict.append(line.replace("\n", ""))
    sort_index.append(len(line.replace("\n", "")))
dict_sort = (np.array(dict)[np.argsort(sort_index)]).tolist()
dict_classify = [[]]
if len(dict_sort) > 0:
    dict_classify[-1].append(dict_sort[0])
for i in range(1, len(dict_sort)):
    sign = 0
    for j in range(len(dict_classify)):
        if dict_classify[j][0] in dict_sort[i]:
            dict_classify[j].append(dict_sort[i])
            continue
        else:
            sign += 1
    if sign == len(dict_classify):
        dict_classify.append([])
        dict_classify[-1].append(dict_sort[i])

for i in range(len(dict_classify)):
    dict_classify[i] = dict_classify[i][::-1]


# 根据字的分类标记将句子分词
def calculate(x, y, id2word, id2tag, res=[]):
    entity = []
    for j in range(len(x)):
        if id2tag[y[j]] == 'B':
            entity = [id2word[x[j]]]
        elif id2tag[y[j]] == 'M' and len(entity) != 0:
            # entity.append(id2word[x[j]] + '/' + id2tag[y[j]])
            entity.append(id2word[x[j]])
        elif id2tag[y[j]] == 'E' and len(entity) != 0:
            entity.append(id2word[x[j]])
            res.append(entity)
            entity = []
        elif id2tag[y[j]] == 'S':
            entity = [id2word[x[j]]]
            res.append(entity)
            entity = []
        else:
            entity = []
    return res


def tokenizer(str):
    # 去除空格
    str.replace(' ', '')
    str_coding = []
    # 未知字标记
    not_contain_sign = 0

    for i in str:
        if word2id.get(i) == None:
            str_coding.append(1)
            not_contain_sign = 1
        else:
            str_coding.append(word2id[i])

    dict_list_find = []
    for i in range(len(dict_classify)):
        dict_list_find.append([])
        for j in range(len(dict_classify[i])):
            temp_index = [r.span() for r in re.finditer(dict_classify[i][j], str)]

            if len(dict_list_find[-1]) <= 0:
                dict_list_find[i] += temp_index
            elif len(temp_index) > 0 and len(dict_list_find[-1]) > 0:
                temp_index_real = []
                for k in range(len(temp_index)):
                    sign_index = 0
                    for m in range(len(dict_list_find[-1])):
                        if temp_index[k][0] >= dict_list_find[-1][m][0] and temp_index[k][1] <= dict_list_find[-1][m][
                            1]:
                            sign_index = 1
                    if sign_index == 0:
                        temp_index_real.append(temp_index[k])
                dict_list_find[-1] += temp_index_real
    # 拉直

    dict_list_find_one = [i for item in dict_list_find for i in item]
    dict_list_find_one_frist = []
    for i in range(len(dict_list_find_one)):
        dict_list_find_one_frist.append(dict_list_find_one[i][0])
    dict_list_find_one_sort_index = np.argsort(dict_list_find_one_frist)
    dict_list_find_one_sort = np.array(dict_list_find_one)[dict_list_find_one_sort_index].tolist()

    # 文本切割
    str_segmentation_by_dict = []
    str_segmentation_by_dict_index = []
    str_dict_sign = []
    predictall = []
    if len(dict_list_find_one_sort) > 0:
        if dict_list_find_one_sort[0][0] - 0 > 0:
            str_segmentation_by_dict_index.append([0, dict_list_find_one_sort[0][0]])
            str_dict_sign.append(0)
        str_segmentation_by_dict_index.append(dict_list_find_one_sort[0])
        str_dict_sign.append(1)
        for i in range(len(dict_list_find_one_sort) - 1):
            str_segmentation_by_dict_index.append([dict_list_find_one_sort[i][1], dict_list_find_one_sort[i + 1][0]])
            str_dict_sign.append(0)
            str_segmentation_by_dict_index.append(dict_list_find_one_sort[i + 1])
            str_dict_sign.append(1)
        if len(str) - dict_list_find_one_sort[-1][1] > 0:
            str_dict_sign.append(0)
            str_segmentation_by_dict_index.append([dict_list_find_one_sort[-1][1], len(str)])
        for i in range(len(str_segmentation_by_dict_index)):
            str_segmentation_by_dict.append(
                str[str_segmentation_by_dict_index[i][0]:str_segmentation_by_dict_index[i][1]])
        for i in range(len(str_segmentation_by_dict)):
            if str_dict_sign[i] == 0:
                strs = re.split(r"([,。()（）：:，-])", str_segmentation_by_dict[i])
                str_split = []
                for i in strs:
                    if len(i) > 0:
                        str_split.append([])
                        for j in i:
                            str_split[-1].append(j)
                str_to_text = []
                for i in str_split:
                    if len(i) >= 1:
                        str_to_text.append([])
                        for j in i:
                            if word2id.get(j) == None:
                                str_to_text[-1].append(1)
                                not_contain_sign = 1
                            else:
                                str_to_text[-1].append(word2id[j])

                for i in range(len(str_to_text)):
                    if str_split[i][0] == ',' or str_split[i][0] == '。' or str_split[i][0] == '(' or str_split[i][
                        0] == ')' or \
                            str_split[i][0] == '（' or str_split[i][0] == '）' or str_split[i][0] == '：' or str_split[i][
                        0] == ':' or \
                            str_split[i][0] == '，' or str_split[i][0] == '-':
                        predictall = predictall + [3]
                    else:
                        START_TAG = "<START>"
                        STOP_TAG = "<STOP>"
                        tag2id[START_TAG] = len(tag2id)
                        tag2id[STOP_TAG] = len(tag2id)
                        entityres = []
                        sentence = torch.tensor(str_to_text[i], dtype=torch.long).to(device)
                        score, predict = model.test(sentence)
                        predictall = predictall + predict

            else:
                predict = []
                if len(str_segmentation_by_dict[i]) <= 1:
                    predict.append(3)
                elif len(str_segmentation_by_dict[i]) <= 2:
                    predict.append(0)
                    predict.append(2)
                else:
                    predict.append(0)
                    for j in range(1, len(str_segmentation_by_dict[i]) - 1):
                        predict.append(1)
                    predict.append(2)
                predictall = predictall + predict

    else:

        strs = re.split(r"([,。()（）：:，-])", str)

        str_split = []
        for i in strs:
            str_split.append([])
            for j in i:
                str_split[-1].append(j)
        str_to_text = []
        for i in str_split:
            if len(i) >= 1:
                str_to_text.append([])
                for j in i:
                    if word2id.get(j) == None:
                        str_to_text[-1].append(1)
                        not_contain_sign = 1
                    else:
                        str_to_text[-1].append(word2id[j])

        predictall = []
        for i in range(len(str_to_text)):
            if str_split[i][0] == ',' or str_split[i][0] == '。' or str_split[i][0] == '(' or str_split[i][0] == ')' or \
                    str_split[i][0] == '（' or str_split[i][0] == '）' or str_split[i][0] == '：' or str_split[i][
                0] == ':' or \
                    str_split[i][0] == '，' or str_split[i][0] == '-':
                predictall = predictall + [3]
            else:
                START_TAG = "<START>"
                STOP_TAG = "<STOP>"
                tag2id[START_TAG] = len(tag2id)
                tag2id[STOP_TAG] = len(tag2id)
                entityres = []
                sentence = torch.tensor(str_to_text[i], dtype=torch.long).to(device)
                score, predict = model.test(sentence)
                predictall = predictall + predict

    sentenceall = torch.tensor(str_coding, dtype=torch.long).to(device)
    entityres = calculate(sentenceall, predictall, id2word, id2tag)
    return entityres, not_contain_sign,


if __name__ == '__main__':
    str = '螺旋桨，客户反应车辆行驶时，打方向有异响，技师检查动力转向中间轴总成不良。'
    res_first, sign = tokenizer(str)
    if sign == 1:
        print('当前句子中有语料库中不包含的字，为了保证分词精度，请您将该预料加入训练集重新训练！')
    # 矫正文字
    str.replace(' ', '')
    res_final = []
    len_sum = 0
    for i in range(len(res_first)):
        len_sum += len(res_first[i])
        res_final.append(str[len_sum - len(res_first[i]): len_sum])
    print(res_final)

