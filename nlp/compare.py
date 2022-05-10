# -*- coding: utf-8 -*-
"""
Created on Mon May  9 21:49:38 2022

@author: Meg
"""
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np




def Company(data1):
    company = set(data1)    
    company_name = []
    company_number = []
    for i in company:
        company_name.append(i)
        company_number.append(data1.count(i))
        
    
    #设置中文字体
    plt.rcParams['font.family'] = ['SimHei']
    
    x = np.random.randint(0,300,127)
    y = np.random.randint(0,300,127)
    z = company_number
    ax = plt.subplot(projection='3d')  # 三维图形
    
    for xx, yy, zz in zip(x,y,z):
        color = np.random.random(3)   # 随机颜色元祖
        ax.bar3d(
            xx,            # 每个柱的x坐标
            yy,            # 每个柱的y坐标
            0,             # 每个柱的起始坐标
            dx=5,          # x方向的宽度
            dy=5,          # y方向的厚度
            dz=zz,         # z方向的高度
            color=color)   #每个柱的颜色
        
    # 设置坐标轴标签    
    ax.set_title('厂家提供主损件数量图')
    ax.set_xlabel('主损件')
    ax.set_ylabel('厂家')
    ax.set_zlabel('数量')
    plt.show()


if __name__=='__main__':
    #导入数据
    data1 = pd.read_excel('../data/汽车数据.xlsx', header=0)['供应商名称'].tolist()
    #画图
    Company(data1)
    
