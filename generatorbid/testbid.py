import abc
from dbutils.pooled_db import PooledDB
import pymysql
from flask import current_app
from core.xmlappcontext import XmlAppContext
import pandas as pd
from generatorbid.bidresource import BidResourceExcel
from generatorbid.bidservice import BidService

class TestBid():
    def __init__(self,gid,month):
        self._gid = gid
        self._month = month
        self.dataresource = BidResourceExcel()

    def getData(self):
        bidRE =  self.dataresource.getData()
        data = {}
        # 市场需求电量
        data['Qd'] = bidRE['市场需求电量-供需比-出清价格'].iloc[1, self._month]
        # 供需比
        data['marketrate'] = bidRE['市场需求电量-供需比-出清价格'].iloc[2, self._month]
        # 出清价格
        data['pmc'] = bidRE['市场需求电量-供需比-出清价格'].iloc[3, self._month]

        # 发电系数
        data['a'] = bidRE['发电商额定功率和发电系数'].iloc[self._gid, 2]
        data['b'] = bidRE['发电商额定功率和发电系数'].iloc[self._gid, 3]
        # data['c'] = bidRE['发电商额定功率和发电系数'].iloc[self._gid, 4]

        # 月利用小时数
        data['TMon'] = bidRE['发电商月利用小时数'].iloc[self._gid, self._month]

        # 发电商月度分解电量
        data['q_YD'] = bidRE['发电商月度基本发电计划'].iloc[self._gid, self._month]

        # 发电商月度剩余发电量
        data['q_Mon'] = bidRE['发电商月度剩余发电量'].iloc[self._gid, self._month]

        return data

if __name__ == '__main__':
    testbid = TestBid(3,3)
    gdata = testbid.getData()
    #print(gdata)
    bidservice = BidService()
    bidservice.service('bidenv-v0',gdata)


