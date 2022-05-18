from py2neo import Graph,Node,Relationship
import pandas as pd
from py2neo import *
import os
graph = Graph(
    "http://localhost:7474",
    username="neo4j",
    password="wmm220601"
)

def Ele_Data(name,data_name):
    count  = 0
    file_path = '../data'
    file_name = name+'.csv'
    path = os.path.join(file_path,file_name)
    frame = pd.read_csv(path, encoding='gbk')

    for i in frame.index:
        '''获取数据'''
        yoga_name = frame[data_name[0]].values[i]
        yoga_ms = frame[data_name[1]].values[i]

        yoga_name = str(yoga_name)
        yoga_ms = str(yoga_ms)

        yoga_node = Node(data_name[2], name=yoga_name)
        ms_node = Node(data_name[3], name=yoga_ms)
        # 创建关系
        yoga_2 = Relationship(yoga_node, name, ms_node)
        try:
            graph.create(yoga_2)
        except:
            continue
        count += 1


    #节点合并
        print(count)


if __name__=='__main__':
    relationship_type = ['生产厂家', '故障部位', '故障等级', '故障类型', '类属关系', '故障时间', '投运时间', '出厂时间', '线路名', '电压等级']
    entity = [['DEP', 'COM', '设备名', '厂家名'], ['DEP', 'PAR', '设备名', '部件名'], ['DEP', 'CLA', '设备名', '故障等级'], \
              ['DEP', 'TYP', '设备名', '故障类型'], ['DEP', 'DEP', '设备名', '设备名'], ['DEP', 'TIM', '设备名', '故障时间'], \
              ['DEP', 'TIM', '设备名', '投运时间'], ['DEP', 'TIM', '设备名', '出厂时间'], ['DEP', 'LIN', '设备名', '线路名'],
              ['LIN', 'VOL', '线路名', '电压等级']]
    for i in range(len(relationship_type)):
        name = relationship_type[i]
        data_name = entity[i]
        Ele_Data(name,data_name)
