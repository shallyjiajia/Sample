# -*- coding: utf-8 -*-
"""
Created on Mon May  9 15:20:57 2022

@author: Meg
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#根据故障进行分类，并统计频率


def Fault(data1):
    fault = set(data1)
    
    type_list = ['指示灯故障','油性故障','操作不良','材质不良','性能故障','外观受损','其他']
    
    fault_light = []
    
    oil = ['油','漏','渗','水']
    fault_oil = []
    
    operatio = ['不动','卡','偏','无法','困难','重','软','有']
    fault_operation = []
    
    texture = ['不良','不佳','断','不平','屏','毛','死','失','音','声','锈','黑']
    fault_texture = []
    
    performance = ['不','翘','断','蹭','路','无','电','烧','热','锁','力','动','风','颠']
    fault_performance = []
    
    apperance = ['漆','皮','裂','凸','凹','磨','变','落','化','铁','破']
    fault_apperance = []
    
    fault_other = []
    for i in fault:
        if '灯' in i:
            fault_light.append(i)
        elif any(k in i for k in oil):
            fault_oil.append(i)
        elif any(k in i for k in operatio):
            fault_operation.append(i)      
        elif any(k in i for k in texture):
            fault_texture.append(i)
        elif any(k in i for k in performance):
            fault_performance.append(i)
        elif any(k in i for k in apperance):
            fault_apperance.append(i)
        else:
            fault_other.append(i)
    fault_len = [len(fault_light),len(fault_oil),len(fault_operation),len(fault_texture),\
      len(fault_performance), len(fault_apperance),len(fault_other)]
    return fault_len
  

def Plot(fault_len):          
    #设置中文字体
    plt.rcParams['font.family'] = ['SimHei']
    
    x1 = ['指示灯故障','油性故障','操作不良','材质不良','性能故障','外观受损','其他']
    y1 = fault_len
    
    for a,b in zip(x1,y1):   #柱子上的数字显示
        plt.text(a,b,'%.0f'%b,ha='center',va='bottom',fontsize=13);
        
    plt.bar(x1,y1,label = '故障分类')
    
    plt.title('绘制柱状图')
    
    plt.ylabel('故障出现频率')
    plt.xlabel('故障属性分类')
    
    plt.legend()
    
    plt.show()
    
if __name__=='__main__':
    #导入数据
    data1 = pd.read_excel('../data/本体字典.xlsx', sheet_name='属性', header=0)['故障A'].tolist()
    #对故障进行分类
    fault_len = Fault(data1)
    #画柱状图
    Plot(fault_len)    
    
