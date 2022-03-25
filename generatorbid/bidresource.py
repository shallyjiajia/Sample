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

    def getData(self):
        data_xls = pd.ExcelFile(self._path)
        data={}
        for name in data_xls.sheet_names:
            df = data_xls.parse(sheet_name=name,header=None)
            data[name]=df
        return data

