# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 15:54:25 2022

@author: Meg
"""

# import pandas as pd
import csv
import pandas as pd
Data = open("更新后词文件.txt",encoding = "utf-8")



def file_word(Data):
    #词集合
    Result = set()
    #语义文本
    File = []
    #文本长度
    lens = []
    for i in range(1000):
        line = Data.readline()
        line = line.split()
        line = list(line)
        File.append(line)
        
        #获取最长词串
        lens.append(len(line))
        
        for i in line:
            Result.add(i)
    Data.close()
    return Result,File,lens


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


Result,File,lens = file_word(Data)
#词向量
Word_Vec = Word_label(Result,File,lens)
#标签
Data_Label = pd.read_excel('汽车数据.xlsx')
Label = Data_Label['健康指数HI']







 
#存入文件
def Word_File(Data_Label,Word_Vec):
    
    #读入标签
    Label = Data_Label['健康指数HI']
    
    #将结果存入csv
    f = open('词向量.csv','w',encoding='utf-8')
    with open('词向量.csv','a',encoding='utf-8',newline='') as f:
        csv_writer = csv.writer(f)
        
        csv_writer.writerow(['X','Y'])
        for i in range(len(Label)):
            # temp = ' '.join(str(i) for i in Word_Vec[i])
            csv_writer.writerow([Word_Vec[i],Label[i]])
    f.close()
    return f

Word_File(Data_Label,Word_Vec) 
    

    

