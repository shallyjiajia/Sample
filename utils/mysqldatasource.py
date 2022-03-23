import pymysql
from dbutils.pooled_db import PooledDB
from utils.classutil import ClassUtil
from xml.dom.minidom import parse
from utils.datasource import DataSource
from datetime import datetime

class MysqlDataSource(DataSource):
    def __init__(self):
        self.appcontext = None
        self.host="localhost"
        self.port = 3306
        self.user=""
        self.password=""
        self.database=""
        self.charset="utf8"
        self.maxconnections = 3
        self.mincached = 1
        self.maxcached = 2
        self.pool = None

    def initialize(self):
        dbconfig = {'host': self.host,
                    'port': self.port,
                    'user': self.user,
                    'password': self.password,
                    'database': self.database,
                    'charset': self.charset
                    }
        pooleddb = PooledDB(
            creator=pymysql,
            maxconnections=self.maxconnections,
            mincached=self.mincached,
            maxcached=self.maxcached,
            maxshared=0,
            blocking=True,
            setsession=[],
            ping=0,
            **dbconfig
        )
        self.pool = pooleddb

    def query(self, sql, param=None):
        """
        @param sql: example : "select code,name,amt1,amt2 from test where code=%s "
        @param param: tuple/list example : ('S2')
        @return: result list
        """
        try :
            dbconn = self.pool.connection()
            cursor = dbconn.cursor()

            if param is None:
                count = cursor.execute(sql)
            else:
                count = cursor.execute(sql, param)
            if count > 0:
                result = cursor.fetchall()
            else:
                result = None
        except Exception as e:
            dbconn.rollback()
        finally:
            cursor.close()
            dbconn.close()

        return result

    def insert(self, sql, value):
        """
        @param sql:[param] example : "insert into test(code,name,amt1,amt2) values(%s,%s,%s,%s)"
        @param value:tuple/list example : ('S6','S6Name',1.9,67)
        """
        try:
            dbconn = self.pool.connection()
            cursor = dbconn.cursor()
            result = cursor.execute(sql, value)
            dbconn.commit()
        except Exception as e:
            dbconn.rollback()
        finally:
            cursor.close()
            dbconn.close()

        return result

    def update(self, sql, param=None):
        """
        @param sql: example : "update test set name=%s,amt1=%s where code=%s"
        @param param:  tuple/list example : ('newname',130,'S3')
        @return: count affected rows
        """
        try :
            dbconn = self.pool.connection()
            cursor = dbconn.cursor()
            cursor.execute(sql, param)
            rowcount = cursor.rowcount
            dbconn.commit()
        except Exception as e:
            dbconn.rollback()
        finally:
            cursor.close()
            dbconn.close()

        return rowcount

    def delete(self, sql, param=None):
        """
        @param sql: example : "delete from test where amt1=%s"
        @param param:  tuple/list example : (100)
        @return: count affected rows
        """
        try :
            dbconn = self.pool.connection()
            cursor = dbconn.cursor()
            cursor.execute(sql, param)
            rowcount = cursor.rowcount
            dbconn.commit()
        except Exception as e:
            dbconn.rollback()
        finally:
            cursor.close()
            dbconn.close()

        return rowcount

if __name__=="__main__":
    beanrepository = {}

    dom = parse("../conf/appcontext.xml")
    document = dom.documentElement
    beandefs = document.getElementsByTagName('bean')
    for beandef in beandefs:
        beanid = beandef.getAttribute('id')
        beancls = beandef.getAttribute('class')
        beaninitmethod = beandef.getAttribute('initmethod')
        print(beanid, beancls, beaninitmethod)

        bean = ClassUtil.newInstance(beancls)

        properties = beandef.getElementsByTagName('property')
        for property in properties:
            ptype = property.getAttribute('type')  # childNodes[0].nodeValue
            pname = property.getAttribute('name')  # childNodes[0].nodeValue
            pvalue = property.getAttribute('value')  # childNodes[0].nodeValue
            if pvalue == "":  # reference
                pvalue = property.getAttribute('ref')
                pref = beanrepository[pvalue]
                setattr(bean, pname, pref)
            else:
                if ptype == "str":
                    setattr(bean, pname, pvalue)
                elif ptype == "int":
                    setattr(bean, pname, int(pvalue))
                elif ptype == "date":
                    setattr(bean, pname, datetime.strptime(pvalue, '%Y-%m-%d'))
                else:
                    setattr(bean, pname, pvalue)

        beanrepository[beanid] = bean
        if beaninitmethod != "":
            im=getattr(bean,beaninitmethod)
            im()

    datasource = beanrepository["datasource"]
    result=datasource.query("select code,name,amt1,amt2 from test where code=%s",('S2'))
    print(result)
    result=datasource.update("update test set name=%s,amt1=%s where code=%s",('newname',88,'S3'))
    print(result)
