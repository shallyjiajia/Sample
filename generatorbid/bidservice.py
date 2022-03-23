from utils.RL.test import A3C
from generatorbid.bid import Bid
from generatorbid.bidresource import BidResourceExcel
from utils.datasource import DataSource
from dbutils.pooled_db import PooledDB
import pymysql

class BidService():
    def __init__(self):
        # self.appcontext = None
        self.datasource = None

    def getData(self):
        bidre = BidResourceExcel()
        return bidre.getData()

    def test(self):
        return self._internal(self.datasource)

    def _internal(self, datasource: DataSource):
        result = datasource.query("select * from test where code='S2'")
        print(result)

        bid = Bid()

        strategy = A3C()
        params = {"Param1": 1, "Param2": "Flop"}

        bid.strategy = strategy
        bid.setParams(params)

        return bid.run()

    def toString(self):
        s = 'This is test bid'
        print(s)
        return s

    def testsql(self):
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
        dbconn = dbpool.connection()
        cursor = dbconn.cursor()
        sql = "select code,name,amt1,amt2 from test where code=%s "
        cursor.execute(sql, ('S2'))
        result = cursor.fetchone()
        print(result)
        cursor.close()
        dbconn.close()

        dbconn = dbpool.connection()
        cursor = dbconn.cursor()
        sql = "insert into test(code,name,amt1,amt2) values(%s,%s,%s,%s)"
        result = cursor.execute(sql, ('S6', 'S6Name', 1.9, 67))
        print(result)
        dbconn.commit()
        cursor.close()
        dbconn.close()

        dbconn = dbpool.connection()
        cursor = dbconn.cursor()
        sql = "update test set name=%s,amt1=%s where code=%s"
        result = cursor.execute(sql, ('newname', 130, 'S3'))
        print(result)
        dbconn.commit()
        cursor.close()
        dbconn.close()

        dbconn = dbpool.connection()
        cursor = dbconn.cursor()
        sql = "delete from test where amt1=%s"
        result = cursor.execute(sql, (100))
        print(result)
        dbconn.commit()
        cursor.close()
        dbconn.close()

if __name__ == "__main__":
    testbid = BidService()
    data=testbid.getData()
    print(data)

    #testbid.testsql()
