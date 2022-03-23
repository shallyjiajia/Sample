import abc
from dbutils.pooled_db import PooledDB
import pymysql
from flask import current_app
from core.xmlappcontext import XmlAppContext
import pandas as pd

class BidResource(metaclass=abc.ABCMeta):
    params={}
    @abc.abstractmethod  # 定义抽象方法，无需实现功能
    def getData(self):
        '子类定义实现获取数据功能'
        pass


class BidResourceMysql(BidResource):
    def __init__(self):
        self._dbpool=self.test()

    def test(self):
        config = {'host': '127.0.0.1',
                  'port': 3306,
                  'user': 'root',
                  'password': 'root',
                  'database': 'electricity',
                  'charset': 'utf8'
                  }
        dbpool = PooledDB(
            creator=pymysql,
            maxconnections=3,
            mincached=1,
            maxcached=2,
            maxshared=0,
            blocking=True,
            setsession=[],
            ping=0,
            **config
        )
        return dbpool


    def context_test(self,context:XmlAppContext):
        self._dbpool = context.beans['dbpool']
        return self.getData()

    #for web mode - @app.route('/api/v1/generatorbid/a3c')
    def web_test(self):
        self._dbpool = current_app.config["context"].beans['dbpool']
        return self.getData()

    def getData(self):
        dbpool = self._dbpool
        dbconn = dbpool.connection()
        cursor = dbconn.cursor()
        cursor.execute("select * from test where code='S1'")
        result = cursor.fetchone()
        print(result)
        dbconn.close()

        return result

class BidResourceExcel(BidResource):
    def __init__(self):
        self._path =r'C:\Users\shally jia\Desktop\远光软件\Sample\data\data.xlsx'
        self._gid=1
        self._month=1
    
    def getData(self):
        bidRE = self.getResource()
        data={}
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

    def getResource(self):
        data_xls = pd.ExcelFile(self._path)
        data={}
        for name in data_xls.sheet_names:
            df = data_xls.parse(sheet_name=name,header=None)
            data[name]=df
        return data

