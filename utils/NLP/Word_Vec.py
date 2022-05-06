# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 15:54:25 2022

@author: Meg
"""
"""
Data格式：data为.txt文件，以空格作为分隔符
file_word函数：返回词的集合，语义文本以及文本长度
Word_label函数：返回词向量
"""
def file_word(Data):
    #词集合
    Result = set()
    #语义文本
    File = []
    #文本长度
    lens = []
    flage = True
    while flage:
        # for i in range(1000):
        line = Data.readline()
        if not line:
            break
        line = line.split()
        line = list(line)
        File.append(line)

        # 获取最长词串
        lens.append(len(line))

        for i in line:
            Result.add(i)
    Data.close()
    return Result, File, lens
def Word_label(Result,File,lens):
    Result_label = list(Result)
    #对每个词进行标记
    Dict_word = {}
    for i in range(len(Result)):
        Dict_word[Result_label[i]] = i+1
    
    MAX = max(lens)
    #对词进行向量化，词的维度为50    
    Word_Vec = []
    for i in range(len(File)):
        temp = []
        for k in File[i]:
            if k in Result_label:
                temp.append(Dict_word[k])
        L = MAX - len(temp)
        if L!=0:
            temp1 = [0 for i in range(L)]
            temp = temp + temp1
        Word_Vec.append(temp)
    return Word_Vec
if __name__=='__main__':
    Data = open("../data/更新后词文件.txt", encoding="utf-8")
    Result,File,lens = file_word(Data)
    Word_Vec = Word_label(Result,File,lens)








 

    

    

