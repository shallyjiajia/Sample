class DataSource():
    def query(self, sql, param=None):
        """
        @param sql: example : "select code,name,amt1,amt2 from test where code=%s "
        @param param: tuple/list example : ('S2')
        @return: result list
        """

    def insert(self, sql, value):
        """
        @param sql:[param] example : "insert into test(code,name,amt1,amt2) values(%s,%s,%s,%s)"
        @param value:tuple/list example : ('S6','S6Name',1.9,67)
        """

    def update(self, sql, param=None):
        """
        @param sql: example : "update test set name=%s,amt1=%s where code=%s"
        @param param:  tuple/list example : ('newname',130,'S3')
        @return: count affected rows
        """

    def delete(self, sql, param=None):
        """
        @param sql: example : "delete from test where amt1=%s"
        @param param:  tuple/list example : (100)
        @return: count affected rows
        """
