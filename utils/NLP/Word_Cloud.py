# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 15:39:49 2022

@author: Meg
"""
        
# import jieba
import wordcloud

import pandas as pd
import numpy as np
na=np.nan



"""
df格式为dataframe
image_path:保存路径和图片名称
"""

def Word_Cloud(df,image_path):
    s = []
    N = df.shape
    for i in range(N[0]):
        for j in range(N[1]):
            flag = pd.isnull(df.loc[i,j])
            if flag !=True :
                s.append(df.loc[i,j])

    text = ' '.join(s) # 连接成字符串

    wc = wordcloud.WordCloud(font_path="msyh.ttc",
                              width = 1000,
                              height = 700,
                              background_color='white',
                              max_words=100)

# msyh.ttc电脑本地字体，写可以写成绝对路径
    wc.generate(text) # 加载词云文本
    wc.to_file(image_path)

if __name__=='__main__':
    df = pd.read_excel(r'../data/故障短语表.xlsx', sheet_name='故障短语', header=None)
    image_path = '../result/故障短语.png'
    Word_Cloud(df,image_path)
