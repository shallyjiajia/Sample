# -*- coding: utf-8 -*-
"""
Created on Mon May  9 21:28:23 2022

@author: Meg
"""

import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np

#根据故障进行分类，并统计频率

def Part(data1):

    part = set(data1)
    
    #将主损件
    type_list = ['灯门锁故障','油箱电机故障','齿轮故障','车内器件','车身装饰','其他']
    
    light = ['灯','座椅','门','锁','烟','储','屏']
    part_light = []
    
    oil = ['油','电','主',]
    part_oil = []
    
    wheel = ['齿','轴','盘','管','车','链']
    part_wheel = []
    
    car = ['空','球','力','动','振','器','手','杆','仪']
    part_car = []
    
    decor = ['嘴','箱','臂','盖','阀','镜','饰','桥','带','风']
    part_decor = []
    
    part_other = []
    
    for i in part:
        if any(k in i for k in light):
            part_light.append(i)
        elif any(k in i for k in oil):
            part_oil.append(i)
        elif any(k in i for k in wheel):
            part_wheel.append(i)
        elif any(k in i for k in car):
            part_car.append(i)
        elif any(k in i for k in decor):
            part_decor.append(i)
        else:
            part_other.append(i)
            
    type_len = [len(part_light),len(part_oil),len(part_wheel),len(part_car),len(part_decor),len(part_other)]
    return type_len

def Plot(type_len):

    #设置中文字体
    plt.rcParams['font.family'] = ['SimHei']
    
    x = [20, 23,  1, 35,  7,  1]
    y = [19, 10,  4, 11, 24, 19]
    z = type_len
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
    ax.set_title('故障分类统计图')
    ax.set_xlabel('故障分类')
    ax.set_ylabel('故障类型')
    ax.set_zlabel('频率')
    
    plt.show()

if __name__=='__main__':
    #导入数据
    data1 = pd.read_excel('../data/汽车数据.xlsx', header=0)['主损件'].tolist()
    #对主损部件数据进行分类并统计数量
    type_len = Part(data1)
    #画图
    Plot(type_len)
    
















